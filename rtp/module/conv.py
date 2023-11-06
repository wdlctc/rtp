from typing import Callable, Optional, Tuple, Any, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import Tensor
from functools import partial
import torch.distributed as dist

from .utils import divide_and_check_no_remainder, affine_weight
from .collectives import gather_from_conv_parallel_region, copy_to_model_parallel_region
from .collectives import set_full_param, set_full_param2, allign_storage, free_storage, _WeightParallelRegion_all, _WeightParallelRegion_test
from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn


class ModelParallelConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        world_size: int = 1,
        rank: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Divide the weight matrix along the last dimension.
        self.world_size = world_size
        self.rank = rank
        self.output_size_per_partition = divide_and_check_no_remainder(out_channels, world_size)

        super(ModelParallelConv2d, self).__init__(in_channels, self.output_size_per_partition, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **factory_kwargs)

    def affine_weight(self, Conv2d_layer):
        if Conv2d_layer is not None:
            affine_weight(self.weight, Conv2d_layer.weight, self.output_size_per_partition, 0, self.world_size, self.rank)
            if hasattr(self, "bias"):
                affine_weight(self.bias, Conv2d_layer.bias, self.output_size_per_partition, 0, self.world_size, self.rank)


class ColumnParallelConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        group=None,
        gather_output = True,
        Conv2d_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ColumnParallelConv2d, self).__init__()

        # Keep input parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gather_output = gather_output

        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(out_channels, self.world_size)

        self.layers = []
        for i in range(self.world_size):
            Conv = ModelParallelConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **factory_kwargs,
                                       world_size = self.world_size, rank = self.rank)

            if i == 0:
                set_full_param(Conv, device, dtype)
                allign_storage(Conv)
                Conv.affine_weight(Conv2d_layer)
            else:
                set_full_param2(Conv, device, dtype, self.layers[0]._full_param)
                allign_storage(Conv)

            self.layers.append(Conv)
        self.conv = self.layers[self.rank]

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore

        output_list = [None for _ in range(self.world_size)]
    
        for i in range(self.world_size):
            index = (self.rank +i) % self.world_size

            input_ = _WeightParallelRegion_before.apply(input_, self.layers[index], i, self)

            output_parallel = self.layers[index](input_)

            output_parallel = _WeightParallelRegion_after.apply(output_parallel, self.layers[index], self.layers[(index+1) % self.world_size], i, self)

            output_list[index] = output_parallel

        output_parallel = torch.cat(output_list, dim=1).contiguous()

        if self.training:
            for i, conv in enumerate(self.layers):
                if i == self.rank:
                    continue
                conv.count = 0
                for p in conv.parameters():
                    p_tmp = p.expand_as(p)
                    assert p_tmp.grad_fn is not None
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    conv.count += 1
                    handle = grad_acc.register_hook(partial(hook_fn, p, conv, self))


        return output_parallel

    def cuda(self, device = None) :
        r"""Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        print('------------------------------')
        print('cuda')
        tmp = self.conv._full_param.detach().clone()

        self._apply(lambda t: t.cuda(device))

        set_full_param(self.conv, self.conv.weight.device, self.conv.weight.dtype)
        allign_storage(self.conv)
        self.conv._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.layers[i], self.layers[self.rank].weight.device, self.layers[self.rank].weight.dtype, self.layers[self.rank]._full_param)
                allign_storage(self.layers[i])


    def to(self, *args, **kwargs):

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        tmp = self.conv._full_param.detach().clone()

        self.conv.to(device=device, dtype=dtype, non_blocking=non_blocking)

        set_full_param(self.conv, self.conv.weight.device, self.conv.weight.dtype)
        allign_storage(self.conv)
        self.conv._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.layers[i], self.layers[self.rank].weight.device, self.layers[self.rank].weight.dtype, self.layers[self.rank]._full_param)
                allign_storage(self.layers[i])


    def set_full_param(self, device=None, dtype=None):
        
        tmp = self.conv._full_param.detach().clone()

        self.conv.to(device=device, dtype=dtype)

        set_full_param(self.conv, self.conv.weight.device, self.conv.weight.dtype)
        allign_storage(self.conv)
        self.conv._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.layers[i], self.layers[self.rank].weight.device, self.layers[self.rank].weight.dtype, self.layers[self.rank]._full_param)
                allign_storage(self.layers[i])