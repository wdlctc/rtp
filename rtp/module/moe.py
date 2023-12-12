# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import copy

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        rank = dist.get_rank(ctx.group)
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate: gate network
        expert: expert network
        group: group to use for all-to-all communication
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)


    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.group, expert_output)
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)

from .collectives import set_full_param, set_full_param2, allign_storage
from .collectives import affine_module_weight, _left_shift_buffer, _right_shift_buffer
from functools import partial


class _WeightParallelRegion_before(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, itr):  # type: ignore
        if itr == 0:
            module._buffer = torch.zeros_like(module.flat_param)
            module.reqs = _left_shift_buffer(module.flat_param.data, module._buffer)
        else:
            for req in module.reqs:
                req.wait()
            module.flat_param.data.copy_(module._buffer)
            if itr != torch.distributed.get_world_size() - 1:
                module.reqs = _left_shift_buffer(module.flat_param.data, module._buffer)
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output, None, None


class _WeightParallelRegion_after(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, module, itr):  # type: ignore
        ctx.module = module
        ctx.itr = itr

        # with torch.cuda.stream(m._streams["rtp"]):
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
                module.reqs = _right_shift_buffer(module.flat_param.data, module._buffer)
        else:
            module._full_grad = torch.zeros_like(module.flat_param)
            module._grad_buffer = torch.zeros_like(module.flat_param)
            module._buffer = torch.zeros_like(module.flat_param)

            for sub_module in module.module_list:
                cur_numel = 0
                for param in sub_module.parameters():
                    param.grad = module._full_grad[cur_numel: cur_numel + param.numel()].view(param.shape)
                    cur_numel += param.numel()

            module.reqs = _right_shift_buffer(module.flat_param.data, module._buffer)
        return grad_output, None, None


class WeightMOELayer(Base):
    def __init__(self, gate: Module, experts: Union[Module, ModuleList], group: Optional[Any] = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)

        self.num_local_experts = len(self.experts)

        param_list = list(experts.parameters())
        self._param_numels = [p.numel() for p in param_list]

        self.flat_param = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in param_list], 0)
        
        cur_numel = 0
        for param in param_list:
            param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
            cur_numel += param.numel()

        self.module_list = []
        
        for i in range(self.world_size):
            if i == self.rank:
                sub_module = self.experts
                self.module_list.append(sub_module)
                continue
            else:
                experts = nn.ModuleList(
                    [copy.deepcopy(expert) for expert in (self.experts)]
                )
                sub_module = experts
                
                cur_numel = 0
                for param in sub_module.parameters():
                    param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
                    cur_numel += param.numel()
            self.module_list.append(sub_module)
        

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)

        chunks = dispatched_input.chunk(self.world_size, dim=0)

        all_outputs = [None for _ in range(self.world_size)]
        for i in range(self.world_size):
            index = (self.rank + i) % self.world_size

            chunk = chunks[index]

            _WeightParallelRegion_before.apply(chunk, self, i)

            experts = self.module_list[index]

            expert_outputs = []
            for expert in experts:
                expert_outputs += [expert(chunk)]
            
            expert_outputs = torch.cat(expert_outputs, dim=1)

            expert_outputs = _WeightParallelRegion_after.apply(expert_outputs,  self, i)

            all_outputs[index] = expert_outputs

        
        all_outputs = torch.stack(all_outputs)
        expert_output = all_outputs.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)

        self._register_post_backward_hooks()
        return combined_output.reshape(input[0].shape)

    def _register_post_backward_hooks(self) -> None:
        for i, sub_module in enumerate(self.module_list):
            if i == self.rank:
                continue
            sub_module.count = 0
            for p in sub_module.parameters():
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    assert p_tmp.grad_fn is not None
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    sub_module.count += 1
                    handle = grad_acc.register_hook(partial(hook_fn, sub_module, self))


    def cuda(self, device=None):
        self.gate.cuda(device)
        self.flat_param = self.flat_param.cuda(device)
        for sub_module in self.module_list:
            cur_numel = 0
            for param in sub_module.parameters():
                param.data = self.flat_param[cur_numel: cur_numel + param.numel()].view(param.shape)
                cur_numel += param.numel()

def hook_fn(sub_module, module, *unused: Any):
    sub_module.count -= 1
    if sub_module.count == 0:
        module.grad_reqs = _right_shift_buffer(module._full_grad.data, module._grad_buffer)
