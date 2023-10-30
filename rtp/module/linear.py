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

from .utils import divide_and_check_no_remainder, affine_weight
from .collectives import gather_from_model_parallel_region, copy_to_model_parallel_region
from .collectives import set_full_param, allign_storage, free_storage, set_full_param2
from .collectives import _WeightParallelRegion_all, _WeightParallelRegion_test

from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn

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
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)
        self.rank = rank
        self.world_size = world_size

        self.linear = ModelParallelLinear(in_features, out_features, bias=bias, **factory_kwargs,
                                          world_size = world_size, rank = rank)
        self.linear.affine_weight(linear_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        output_parallel = self.linear(input_parallel)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

class WeightParallelLinear(torch.nn.Module):
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
        gather_weight: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
        linear_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_weight = gather_weight
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)
        self.world_size = world_size
        self.rank = rank

        self.linears = []
        for i in range(self.world_size):
            linear = ModelParallelLinear(in_features, out_features, bias=bias, **factory_kwargs,
                                         world_size = world_size, rank = rank)
            if i == 0:
                set_full_param(linear, device, dtype)
                allign_storage(linear)
                linear.affine_weight(linear_layer)
            else:
                set_full_param2(linear, device, dtype, self.linears[0]._full_param)
                allign_storage(linear)
            self.linears.append(linear)

        self.linear = self.linears[self.rank]
        
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

