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

import torch
from typing import Any
from .utils import divide_and_check_no_remainder, divide, split_tensor_along_last_dim

def set_full_param(module, device, dtype):
    
    factory_kwargs = {'device': device, 'dtype': dtype}
    total_numel = 0
    for param_name, param in module.named_parameters():
        total_numel += param.data.numel()
        param._numel = param.data.numel()
        param._shape = param.shape
    
    module.total_numel = total_numel
    module._full_param = torch.zeros(total_numel, **factory_kwargs)
    
    cur_numel = 0
    for param_name, param in module.named_parameters():
        module._full_param[cur_numel: cur_numel + param._numel].copy_(param.data.view(-1))
        param.data.storage().resize_(0)
        cur_numel += param._numel



def affine_module_weight(module, original_module):
    
    for param, original_param in zip(module.parameters(), original_module.parameters()):
        param.data.copy_(original_param.data)
        

def set_full_param2(module, device, dtype, full_param):
    
    factory_kwargs = {'device': device, 'dtype': dtype}
    total_numel = 0
    for param_name, param in module.named_parameters():
        total_numel += param.data.numel()
        param._numel = param.data.numel()
        param._shape = param.shape
        param.data.storage().resize_(0)
    
    module.total_numel = total_numel
    module._full_param = full_param


def allign_storage(module):
    cur_numel = 0
    for param_name, param in module.named_parameters():
        param.data = module._full_param[cur_numel: cur_numel + param._numel].view(param._shape)
        cur_numel += param._numel

def free_storage(module):
    for param_name, param in module.named_parameters():
        param.data.storage().resize_(0)
    module._full_param.storage().resize_(0)

def free_grad(module):
    for param_name, param in module.named_parameters():
        module._full_grad.storage().resize_(0)
    module._full_param.storage().resize_(0)

def allign_grad(module):
    cur_numel = 0
    for param_name, param in module.named_parameters():
        param.grad = module._full_grad[cur_numel: cur_numel + param._numel].view(param.shape)
        cur_numel += param._numel


def _right_shift_copy(input_, output):

    group = torch.distributed.distributed_c10d._get_default_group()

    # torch.distributed.barrier()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    recv_buff = torch.zeros_like(input_)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank + 1)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_buff, (rank - 1 + world_size)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    output.data.copy_(recv_buff)
    recv_buff = None

def _left_shift_copy(input_, output):

    group = torch.distributed.distributed_c10d._get_default_group()

    # torch.distributed.barrier()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    recv_buff = torch.zeros_like(input_)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - 1 + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_buff, (rank +1)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    output.data.copy_(recv_buff)
    recv_buff = None


class _WeightParallelRegion_moe(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, next_module, itr):  # type: ignore
        ctx.module = module
        ctx.next_module = next_module
        ctx.itr = itr
        if itr != torch.distributed.get_world_size() - 1:
            _right_shift_copy(module._full_param.data, next_module._full_param)

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        next_module = ctx.next_module
        itr = ctx.itr

        if itr != torch.distributed.get_world_size() - 1:
            module._full_grad = next_module._full_grad
            _left_shift_copy(next_module._full_param.data, module._full_param)
            _left_shift_copy(next_module._full_grad.data, module._full_grad)
            allign_grad(module)
            
        else:
            module._full_grad = torch.zeros_like(module._full_param)
            allign_grad(module)

        return grad_output, None, None, None

class _WeightParallelRegion_test(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, next_module, itr):  # type: ignore
        ctx.module = module
        ctx.next_module = next_module
        ctx.itr = itr
        if itr != torch.distributed.get_world_size() - 1:
            next_module._full_param.data.copy_(_right_shift(module._full_param.data))

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        next_module = ctx.next_module
        itr = ctx.itr

        if itr != torch.distributed.get_world_size() - 1:
            module._full_param.data.copy_(_left_shift(next_module._full_param.data))
            module._full_grad = _left_shift(next_module._full_grad.data)
            allign_grad(module)
            
        else:
            module._full_grad = torch.zeros_like(module._full_param)
            allign_grad(module)

        return grad_output, None, None, None

def hook_fn(p, layer, module, *unused: Any):
    layer.count -= 1

    if layer.count == 0:
        module.grad_reqs = _right_shift_buffer(layer._full_grad.data, layer._grad_buffer)

def _right_shift_buffer(input_, buffer):

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

def _left_shift_buffer(input_, buffer):

    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - 1 + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, buffer, (rank +1)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    return reqs

class _WeightParallelRegion_before(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, itr, m):  # type: ignore
        if itr == 0:
            for param in module.parameters():
                if param.device != module._full_param.device:
                    m.set_full_param(param.device, module.weight.dtype)
            module._buffer = torch.zeros_like(module._full_param)
            m.reqs = _left_shift_buffer(module._full_param.data, module._buffer)
        else:
            for req in m.reqs:
                req.wait()
            module._full_param.data.copy_(module._buffer)
            if itr != torch.distributed.get_world_size() - 1:
                m.reqs = _left_shift_buffer(module._full_param.data, module._buffer)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output, None, None, None


class _WeightParallelRegion_after(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, next_module, itr, m):  # type: ignore
        ctx.module = module
        ctx.next_module = next_module
        ctx.itr = itr
        ctx.m = m

        # with torch.cuda.stream(m._streams["rtp"]):
        if itr != torch.distributed.get_world_size() - 1:
            next_module._buffer = module._buffer
        module._buffer = None

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        next_module = ctx.next_module
        itr = ctx.itr
        m = ctx.m

        if itr != torch.distributed.get_world_size() - 1:

            for req in m.reqs:
                req.wait()
            for req in m.grad_reqs:
                req.wait()
            module._full_param.data.copy_(next_module._buffer)
            module._full_grad = next_module._full_grad
            module._full_grad.data.copy_(next_module._grad_buffer)
            allign_grad(module)

            if itr != 0:
                module._grad_buffer = next_module._grad_buffer
                module._buffer = next_module._buffer
                m.reqs = _right_shift_buffer(module._full_param.data, module._buffer)

            next_module._full_grad = None
            next_module._grad_buffer = None
            next_module._buffer = None
            


        else:
            module._full_grad = torch.zeros_like(module._full_param)
            module._grad_buffer = torch.zeros_like(module._full_param)
            module._buffer = torch.zeros_like(module._full_param)
            allign_grad(module)
            m.reqs = _right_shift_buffer(module._full_param.data, module._buffer)

        return grad_output, None, None, None, None

class _WeightParallelRegion_all(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, next_module, itr):  # type: ignore
        ctx.module = module
        ctx.next_module = next_module
        ctx.itr = itr
        if itr != torch.distributed.get_world_size() - 1:
            next_module._full_param.data.storage().resize_(next_module.total_numel)
            next_module._full_param.data.copy_(_right_shift(module._full_param.data))
            allign_storage(next_module)
            free_storage(module)

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        next_module = ctx.next_module
        itr = ctx.itr

        if itr != torch.distributed.get_world_size() - 1:
            module._full_param.data.storage().resize_(next_module.total_numel)
            module._full_param.data.copy_(_left_shift(next_module._full_param.data))
            module._full_grad = next_module._full_grad
            module._full_grad.data.copy_(_left_shift(next_module._full_grad.data))
            free_storage(next_module)
            free_grad(next_module)
            allign_storage(module)
            allign_grad(module)
            
        else:
            module._full_grad = torch.zeros_like(module._full_param)
            allign_grad(module)

        return grad_output, None, None, None


class _WeightParallelRegion_weight(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, module, itr):  # type: ignore
        ctx.module = module
        ctx.itr = itr
        if itr != torch.distributed.get_world_size() - 1:
            module._full_param.data = _right_shift(module._full_param.data)
            allign_storage(module)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output, None

def _right_shift(input_):

    group = torch.distributed.distributed_c10d._get_default_group()

    # torch.distributed.barrier()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    recv_buff = torch.zeros_like(input_)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank + 1)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_buff, (rank - 1 + world_size)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    # torch.distributed.barrier()

    return recv_buff

def _left_shift(input_):

    group = torch.distributed.distributed_c10d._get_default_group()

    # torch.distributed.barrier()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    recv_buff = torch.zeros_like(input_)

    send_op = torch.distributed.P2POp(torch.distributed.isend, input_, (rank - 1 + world_size)%world_size)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, recv_buff, (rank +1)%world_size)

    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    # torch.distributed.barrier()

    return recv_buff


class _GatherFromConvParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        group = torch.distributed.distributed_c10d._get_default_group()

        # Bypass the function if we are using only 1 GPU.
        if torch.distributed.get_world_size(group=group) == 1:
            return input_

        # Size and dimension.
        last_dim = 1
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=last_dim).contiguous()

        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        group = torch.distributed.distributed_c10d._get_default_group()

        # Bypass the function if we are using only 1 GPU.
        if torch.distributed.get_world_size(group=group) == 1:
            return grad_output

        # Split along last dimension.
        world_size = torch.distributed.get_world_size(group=group)
        # Get the size and dimension.
        last_dim =1
        last_dim_size = divide(grad_output.size()[last_dim], world_size)
        # Split.
        tensor_list = torch.split(grad_output, last_dim_size, dim=last_dim)

        # Note: torch.split does not create contiguous tensors by default.
        rank = torch.distributed.get_rank(group=group)
        output = tensor_list[rank].contiguous()

        return output

def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output

def _reduce(ctx, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = torch.distributed.distributed_c10d._get_default_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)

class _ShiftToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _right_shift(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _left_shift(grad_output)

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)

def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)

def gather_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)

def gather_from_conv_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromConvParallelRegion.apply(input_)

def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)

def shift_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ShiftToModelParallelRegion.apply(input_)

class _WeightParallelRegion_attention(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, next_module, itr):  # type: ignore
        ctx.module = module
        ctx.next_module = next_module
        ctx.itr = itr
        if itr != torch.distributed.get_world_size() - 1:
            next_module._full_param.data = _right_shift(module._full_param.data)
            next_module.allign_storage()
            # module.free_storage()

        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        module = ctx.module
        next_module = ctx.next_module
        itr = ctx.itr

        if itr != torch.distributed.get_world_size() - 1:
            module._full_param.data = _left_shift(next_module._full_param.data)
            module._full_grad = _left_shift(next_module._full_grad.data)
            next_module.free_storage()
            next_module.free_grad()
            module.allign_storage()
            module.allign_grad()
        else:
            module._full_grad = torch.zeros_like(module._full_param)
            module.allign_grad()


        return grad_output, None, None, None


def zero_full_grads(module):
    """
    Recursively replace all linear layers in the module with the custom layer.
    
    Args:
    - module (nn.Module): The module (or model) to modify.
    - custom_layer_class (nn.Module): The custom layer class to replace with.
    """
    for name, child in module.named_children():
        if hasattr(module, "_full_grad"):
            module._full_grad = None
        else:
            zero_full_grads(child)
