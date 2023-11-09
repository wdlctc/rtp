# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from .collectives import set_full_param, allign_storage, set_full_param2

from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn
from .utils import divide_and_check_no_remainder, split_tensor_along_last_dim

class ModelParallelLinear(torch.nn.Linear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
        linear_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Divide the weight matrix along the last dimension.
        self.world_size = world_size
        self.rank = rank
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        super(ModelParallelLinear, self).__init__(in_features, self.output_size_per_partition, bias, **factory_kwargs)


    def affine_weight(self, linear_layer):
        if linear_layer is not None:
            affine_weight(self.weight, linear_layer.weight, self.output_size_per_partition, 0, self.world_size, self.rank)
            if hasattr(self, "bias"):
                affine_weight(self.bias, linear_layer.bias, self.output_size_per_partition, 0, self.world_size, self.rank)

class ModelParallelLinear2(torch.nn.Linear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
        linear_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Divide the weight matrix along the last dimension.
        self.world_size = world_size
        self.rank = rank
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, world_size)

        super(ModelParallelLinear2, self).__init__(self.input_size_per_partition, out_features, bias, **factory_kwargs)


    def affine_weight(self, linear_layer):
        if linear_layer is not None:
            affine_weight(self.weight, linear_layer.weight, self.input_size_per_partition, 1, self.world_size, self.rank)
            if hasattr(self, "bias"):
                self.bias.data.copy_(linear_layer.bias.data)
                self.bias.data.div_(self.world_size)

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_weight: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
        group=None,
        linear_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_weight = gather_weight
        
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, self.world_size)

        self.linears = []
        for i in range(self.world_size):
            linear = ModelParallelLinear(in_features, out_features, bias=bias, **factory_kwargs,
                                         world_size = self.world_size, rank = self.rank)
            if i == 0:
                set_full_param(linear, device, dtype)
                allign_storage(linear)
                linear.affine_weight(linear_layer)
            else:
                set_full_param2(linear, device, dtype, self.linears[0]._full_param)
                allign_storage(linear)
            self.linears.append(linear)

        self.linear = self.linears[self.rank]
        # print('----------------------------')
        # print(self.linear._full_param)
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = input_
        # Matrix multiply.

        output_list = [None for _ in range(self.world_size)]

        for i in range(self.world_size):
            index = (self.rank +i) % self.world_size

            input_parallel = _WeightParallelRegion_before.apply(input_parallel, self.linears[index], i, self)

            output = self.linears[index](input_parallel)

            output = _WeightParallelRegion_after.apply(output, self.linears[index], self.linears[(index+1) % self.world_size], i, self)

            output_list[index] = output

        output_parallel = torch.cat(output_list, dim=-1).contiguous()

        if self.training:
            for i, linear in enumerate(self.linears):
                if i == self.rank:
                    continue
                linear.count = 0
                for p in linear.parameters():
                    p_tmp = p.expand_as(p)
                    assert p_tmp.grad_fn is not None
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    linear.count += 1
                    handle = grad_acc.register_hook(partial(hook_fn, p, linear, self))

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
        tmp = self.linear._full_param.detach().clone()

        self._apply(lambda t: t.cuda(device))

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])


    def to(self, *args, **kwargs):

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        tmp = self.linear._full_param.detach().clone()

        self.linear.to(device=device, dtype=dtype, non_blocking=non_blocking)

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])

    def set_full_param(self, device=None, dtype=None):

        # print('--------------------------------------------')
        # print(self.linear._full_param)
        tmp = self.linear._full_param.detach().clone()

        self.linear.to(device=device, dtype=dtype)

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])

class RowParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_weight: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
        group=None,
        linear_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_weight = gather_weight
        
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, self.world_size)

        self.linears = []
        for i in range(self.world_size):
            linear = ModelParallelLinear2(in_features, out_features, bias=bias, **factory_kwargs,
                                         world_size = self.world_size, rank = self.rank)
            if i == 0:
                set_full_param(linear, device, dtype)
                allign_storage(linear)
                linear.affine_weight(linear_layer)
            else:
                set_full_param2(linear, device, dtype, self.linears[0]._full_param)
                allign_storage(linear)
            self.linears.append(linear)

        self.linear = self.linears[self.rank]
        # print('----------------------------')
        # print(self.linear._full_param)
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_list = split_tensor_along_last_dim(input_, self.world_size)
        # Matrix multiply.

        for i in range(self.world_size):
            index = (self.rank +i) % self.world_size

            input_parallel = input_list[index]
            input_parallel = _WeightParallelRegion_before.apply(input_parallel, self.linears[index], i, self)

            output = self.linears[index](input_parallel)

            output = _WeightParallelRegion_after.apply(output, self.linears[index], self.linears[(index+1) % self.world_size], i, self)

            if i == 0:
                output_parallel = output
            else:
                output_parallel = output + output_parallel


        if self.training:
            for i, linear in enumerate(self.linears):
                if i == self.rank:
                    continue
                linear.count = 0
                for p in linear.parameters():
                    p_tmp = p.expand_as(p)
                    assert p_tmp.grad_fn is not None
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    linear.count += 1
                    handle = grad_acc.register_hook(partial(hook_fn, p, linear, self))

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
        tmp = self.linear._full_param.detach().clone()

        self._apply(lambda t: t.cuda(device))

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])


    def to(self, *args, **kwargs):

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        tmp = self.linear._full_param.detach().clone()

        self.linear.to(device=device, dtype=dtype, non_blocking=non_blocking)

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])

    def set_full_param(self, device=None, dtype=None):

        # print('--------------------------------------------')
        # print(self.linear._full_param)
        tmp = self.linear._full_param.detach().clone()

        self.linear.to(device=device, dtype=dtype)

        set_full_param(self.linear, self.linear.weight.device, self.linear.weight.dtype)
        allign_storage(self.linear)
        self.linear._full_param.data.copy_(tmp.data)
        tmp = None

        for i in range(self.world_size):
            if i != self.rank:
                set_full_param2(self.linears[i], self.linears[self.rank].weight.device, self.linears[self.rank].weight.dtype, self.linears[self.rank]._full_param)
                allign_storage(self.linears[i])
