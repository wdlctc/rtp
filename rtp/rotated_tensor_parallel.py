import torch
import torch.nn as nn
import torch.distributed as dist
import typing
import copy
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from .attention import ParallelMultiheadAttention

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

ParamGroups = Optional[Union[List[List[nn.Parameter]], List[nn.Parameter]]]

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, dim: int = -1
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    dim_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _clock_rotation_buffer(input_, buffer):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank + 1)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank - 1 + world_size)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs


def counter_clock_rotation_buffer(input_, buffer):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - 1 + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank + 1)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs


def hook_fn(sub_module, module, *unused: Any):
    sub_module.count -= 1
    if sub_module.count == 0:
        module.grad_reqs = counter_clock_rotation_buffer(module._full_grad.data, module._grad_buffer)


def allign_grad(module):
    cur_numel = 0
    for param_name, param in module.named_parameters():
        param.grad = module._full_grad[cur_numel: cur_numel + param._numel].view(param.shape)
        cur_numel += param._numel

class ParallelRegion_before(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, itr):  # type: ignore
        if itr == 0:
            module._buffer = torch.zeros_like(module.flat_param)
            module.reqs = _clock_rotation_buffer(module.flat_param.data, module._buffer)
        else:
            for req in module.reqs:
                req.wait()
            module.flat_param.data.copy_(module._buffer)
            module.reqs = _clock_rotation_buffer(module.flat_param.data, module._buffer)

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output, None, None



class ParallelRegion_after(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, itr):  # type: ignore
        ctx.module = module
        ctx.itr = itr
        if itr == torch.distributed.get_world_size() - 1:
            module._buffer = None

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        itr = ctx.itr

        if itr != torch.distributed.get_world_size() - 1:
            for req in module.reqs:
                req.wait()
            for req in module.grad_reqs:
                req.wait()
            module.flat_param.data.copy_(module._buffer)
            module._full_grad.data.copy_(module._grad_buffer)

            if itr == 0:
                module._full_grad = None
                module._grad_buffer = None
                module._buffer = None
            else:
                module.reqs = counter_clock_rotation_buffer(module.flat_param.data, module._buffer)
        else:
            module._full_grad = torch.zeros_like(module.flat_param)
            module._grad_buffer = torch.zeros_like(module.flat_param)
            module._buffer = torch.zeros_like(module.flat_param)

            for sub_module in module.module_list:
                cur_numel = 0
                for param in sub_module.parameters():
                    param.grad = module._full_grad[cur_numel: cur_numel + param.numel()].view(param.shape)
                    cur_numel += param.numel()

            module.reqs = counter_clock_rotation_buffer(module.flat_param.data, module._buffer)
        return grad_output, None, None


class FlattenParamsWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        param_list: ParamGroups = None,
        flat_param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Handle param_list being None.
        if param_list is None:
            param_list = list(module.parameters())

        self.total_numel = 0
        for param in param_list:
            self.total_numel += param.data.numel()
            param._numel = param.data.numel()
            param._shape = param.shape

        self.flat_param = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in param_list], 0)
        self.flat_param = nn.Parameter(self.flat_param, requires_grad=param_list[0].requires_grad)

class FlyweightWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
        param_list: ParamGroups = None,
        flat_param_names: Optional[List[str]] = None,
        partition_dim: int = None, 
        output_partition_dim: int = None,
        row_partition: bool = False,
        input_partition_dim: int = None,
        cat_output: bool = True,
    ):
        super().__init__()
        self.module = module

        if partition_dim is None:
            partition_dim = -1
        self.partition_dim = partition_dim
        if output_partition_dim is None:
            output_partition_dim = -1
        if input_partition_dim is None:
            input_partition_dim = -1
        self.output_partition_dim = output_partition_dim
        self.input_partition_dim = input_partition_dim
        self.row_partition = row_partition  
        self.cat_output = cat_output
        self.FlyweightList = []

        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
        # Handle param_list being None.
        if param_list is None:
            param_list = list(module.parameters())

        self._param_numels = [p.numel() for p in param_list]

        self.flat_param = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in param_list], 0)
        # self.flat_param = nn.Parameter(self.flat_param, requires_grad=param_list[0].requires_grad)

        cur_numel = 0
        for param in param_list:
            param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
            cur_numel += param.numel()

        self.module_list = []
        for i in range(self.world_size):
            if i == self.rank:
                sub_module = self.module
                self.module_list.append(sub_module)
                continue
            sub_module = copy.deepcopy(self.module)

            cur_numel = 0
            for param in sub_module.parameters():
                param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
                cur_numel += param.numel()

            self.module_list.append(sub_module)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:

        if self.row_partition:
            input_list = split_tensor(args[0], self.world_size, dim=self.input_partition_dim)
            
            for i in range(self.world_size):
                index = (self.rank -i +self.world_size) % self.world_size
                input_parallel = input_list[index]
                ParallelRegion_before.apply(input_parallel, self, i)
                output = self.module_list[index](input_parallel)
                output = ParallelRegion_after.apply(output, self, i)
                
                if i == 0:
                    output_parallel = output
                else:
                    output_parallel = output + output_parallel
        else:
            output_list = [None for _ in range(self.world_size)]
            
            for i in range(self.world_size):
                index = (self.rank -i +self.world_size) % self.world_size
                ParallelRegion_before.apply(args[0], self, i)
                output = self.module_list[index](*args, **kwargs)
                output = ParallelRegion_after.apply(output, self, i)
                if self.cat_output:
                    output_list[index] = output
                else:
                    if i == 0:
                        output_parallel = output
                    else:
                        output_parallel = output + output_parallel

            if self.cat_output:
                output_parallel = torch.cat(output_list, dim=self.output_partition_dim).contiguous()
        
        return output_parallel

class RotatedTensorParallel(nn.Module):
    def __init__(
        self,
        module,
        group: Optional[Any] = None,):
        super().__init__()
        
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.module = module
        self.FlyweightModule_list = []
        self.RecursiveVisit('module', self.module, self)

    def RecursiveVisit(self, name, module, upper_module):
        """
        Recursively replace layers in the module with the custom layer.
        
        Args:
        - module (nn.Module): The module (or model) to modify.
        - custom_layer_class (nn.Module): The custom layer class to replace with.
        """
            
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_MultiheadAttention = isinstance(module, nn.MultiheadAttention)

        if has_child and not is_MultiheadAttention:
            for name, child in module.named_children():
                self.RecursiveVisit(name, child, module)
        else:
            if has_parameters:
                pass
                if isinstance(module, nn.Linear):
                    if module.out_features % self.world_size == 0:
                        module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=0)[self.rank])
                        if module.bias is not None:
                            module.bias = nn.Parameter(split_tensor(module.bias, self.world_size, dim=0)[self.rank])
                        module = FlyweightWarpper(module, self.group)
                    elif module.in_features % self.world_size == 0:
                        module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=-1)[self.rank])
                        if module.bias is not None:
                            module.bias.data.div_(self.world_size)
                        module = FlyweightWarpper(module, self.group, row_partition=True, input_partition_dim=-1)
                    else:
                        raise ValueError("The input or output features of the linear layer must be divisible by the world size.")
                    
                    setattr(upper_module, name, module)

                    self.FlyweightModule_list.append(module)

                elif isinstance(module, nn.Conv2d):
                    if module.out_channels % self.world_size == 0:
                        module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=0)[self.rank])
                        if module.bias is not None:
                            module.bias = nn.Parameter(split_tensor(module.bias, self.world_size, dim=0)[self.rank])
                    else:
                        raise ValueError("The input or output channels of the conv layer must be divisible by the world size.")
                    module = FlyweightWarpper(module, self.group, output_partition_dim=1)
                    
                    setattr(upper_module, name, module)

                    self.FlyweightModule_list.append(module)
                elif isinstance(module, nn.Embedding):
                    if module.embedding_dim % self.world_size == 0:
                        module.weight = nn.Parameter(split_tensor(module.weight, self.world_size, dim=1)[self.rank])
                        module = FlyweightWarpper(module, self.group)
                        setattr(upper_module, name, module)
                        self.FlyweightModule_list.append(module)
                    else:
                        raise ValueError("The embedding_dim of the embedding layer must be divisible by the world size.")
                elif isinstance(module, nn.MultiheadAttention):

                    embed_dim=module.embed_dim
                    num_heads=module.num_heads
                    dropout=module.dropout
                    bias=module.in_proj_bias is not None
                    add_bias_kv=module.bias_k is not None
                    add_zero_attn=module.add_zero_attn
                    kdim=module.kdim
                    vdim=module.vdim

                    device = module.in_proj_weight.device
                    dtype = module.in_proj_weight.dtype

                    r = ParallelMultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias,
                                    add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                    kdim=kdim, vdim=vdim,
                                    device=device, dtype=dtype)

                    index = [self.rank, self.rank+self.world_size, self.rank+self.world_size*2]
                    for n, param in module.named_parameters():
                        if n == 'in_proj_weight':
                            weight_list = split_tensor(param, self.world_size*3, dim=0)
                            weight_list = [weight_list[i] for i in index]
                            weight = nn.Parameter(torch.cat(weight_list, dim=0).contiguous())
                            setattr(r, n, weight)
                        if n == 'in_proj_bias':
                            bias_list = split_tensor(param, self.world_size*3, dim=0)
                            bias_list = [bias_list[i] for i in index]
                            bias = nn.Parameter(torch.cat(bias_list, dim=0).contiguous())
                            setattr(r, n, bias)
                        if n == 'out_proj.weight':
                            r.out_proj.weight = nn.Parameter(split_tensor(module.out_proj.weight, self.world_size, dim=1)[self.rank])
                        if n == 'out_proj.bias':
                            r.out_proj.bias.data.div_(self.world_size)
                    setattr(upper_module, name, r)
                    r = FlyweightWarpper(r, self.group, cat_output=False)
                    self.FlyweightModule_list.append(r)
                # elif isinstance(module, nn.LayerNorm):
                #     pass
                # else:
                #     print(module)
                #     raise ValueError("The layer must be nn.Linear, nn.Conv2d or nn.Embedding.")
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        self._register_post_backward_hooks()
        return outputs

    def _register_post_backward_hooks(self) -> None:
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        for module in self.FlyweightModule_list:
            for i, sub_module in enumerate(module.module_list):
                if i == self.rank:
                    continue
                sub_module.count = 0
                for p in sub_module.parameters():
                    if p.requires_grad:
                        # Register a hook.
                        p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                        assert p_tmp.grad_fn is not None
                        grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                        sub_module.count += 1
                        handle = grad_acc.register_hook(partial(hook_fn, sub_module, module))

    def cuda(self, device=None):
        self.module.cuda(device)
        for module in self.FlyweightModule_list:
            module.cuda(device)
            module.flat_param = module.flat_param.cuda(device)
            for sub_module in module.module_list:
                cur_numel = 0
                for param in sub_module.parameters():
                    param.data = module.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
                    cur_numel += param.numel()