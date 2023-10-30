# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from .utils import affine_weight
from .utils import divide_and_check_no_remainder, affine_weight
from .collectives import set_full_param, allign_storage, set_full_param2
from .collectives import _WeightParallelRegion_test

gumbel_map: Dict[torch.device, Callable] = {}


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(tensor: torch.Tensor, num_classes: int) -> Tensor:
    """Workaround for https://github.com/pytorch/pytorch/issues/55579"""
    assert num_classes > 0, "num_classes must be a positive integer"
    ret = torch.zeros(tensor.shape + (num_classes,), device=tensor.device, dtype=tensor.dtype)
    ret.scatter_(-1, tensor.unsqueeze(-1), 1)
    return ret


def top2gating(logits: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # NOTE(msb) softmax requires FP32: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
    gates = F.softmax(logits, dim=1, dtype=torch.float)

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # capacity = 2S/E
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    gates1_s = (gates * mask1).sum(dim=1)  # einsum("se,se->s")
    gates2_s = (gates * mask2).sum(dim=1)  # einsum("se,se->s")
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2  # einsum("s,se->se")
    locations1_sc = one_hot(locations1_s, num_classes=capacity)
    locations2_sc = one_hot(locations2_s, num_classes=capacity)
    combine1_sec = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine2_sec = gates2.unsqueeze(2) * locations2_sc.unsqueeze(1)  # einsum("se,sc->sec")
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux.to(logits.dtype), combine_weights.to(logits.dtype), dispatch_mask



class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = self.wg(input)
        return top2gating(logits)

class WeightTop2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        gate = None,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.num_experts = num_experts
        self.world_size = torch.distributed.get_world_size()
        self.rank  = torch.distributed.get_rank()
        self.num_local_experts = divide_and_check_no_remainder(num_experts,  self.world_size)
        self.linears = []
        # self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.wgs = []

        for i in range(self.world_size):
            wg = torch.nn.Linear(model_dim, self.num_local_experts, bias=False).to(device)
            if i == 0:
                set_full_param(wg, device, dtype)
                allign_storage(wg)
                self.affine_weight(wg, gate)
            else:
                set_full_param2(wg, device, dtype, self.wgs[0]._full_param)
                allign_storage(wg)
            self.wgs.append(wg)

        
        self.wg = self.wgs[self.rank]


    def affine_weight(self, wg, layer):
        if layer is not None:
            affine_weight(wg.weight, layer.wg.weight, self.num_local_experts, 0, self.world_size, self.rank)
 

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        output_list = [None for _ in range(self.world_size)]
        for i in range(self.world_size):
            index = (self.rank +i) % self.world_size

            output = self.wgs[index](input)

            output = _WeightParallelRegion_test.apply(output, self.wgs[index], self.wgs[(index+1) % self.world_size], i)

            output_list[index] = output
        logits = torch.cat(output_list, dim=-1).contiguous()
        return top2gating(logits)