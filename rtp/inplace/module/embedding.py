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

from .utils import divide_and_check_no_remainder, affine_weight
from .collectives import gather_from_model_parallel_region, copy_to_model_parallel_region
from .collectives import set_full_param, set_full_param2, allign_storage, free_storage, _WeightParallelRegion_all, _WeightParallelRegion_test



class ModelParallelEmbedding(torch.nn.Embedding):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Divide the weight matrix along the last dimension.
        self.world_size = world_size
        self.rank = rank
        self.embedding_dim_per_partition = divide_and_check_no_remainder(embedding_dim, world_size)

        super(ModelParallelEmbedding, self).__init__(num_embeddings, self.embedding_dim_per_partition, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **factory_kwargs)

    def affine_weight(self, Embedding_layer):
        if Embedding_layer is not None:
            affine_weight(self.weight, Embedding_layer.weight, self.embedding_dim_per_partition, 1, self.world_size, self.rank)


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
        Embedding_layer = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)

        self.world_size = world_size
        self.rank = rank

        # Allocate weights.
        self.Embedding = ModelParallelEmbedding(num_embeddings, embedding_dim, world_size, rank, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, init_method, keep_master_weight_for_test, 
                                                **factory_kwargs)

        self.Embedding.affine_weight(Embedding_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_parallel = input_
        output_parallel = self.Embedding(input_parallel)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class WeightParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        world_size: int,
        rank: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
        Embedding_layer = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        self.world_size = world_size
        self.rank = rank
        # Divide the weight matrix along the embedding dimension.
        self.embedding_dim_per_partition = divide_and_check_no_remainder(self.embedding_dim, world_size)

        self.embeddings = []
        for i in range(self.world_size):
            Embedding = ModelParallelEmbedding(num_embeddings, embedding_dim, world_size, rank, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, init_method, keep_master_weight_for_test, 
                                                **factory_kwargs)

            if i == 0:
                set_full_param(Embedding, device, dtype)
                allign_storage(Embedding)
                Embedding.affine_weight(Embedding_layer)
            else:
                set_full_param2(Embedding, device, dtype, self.embeddings[0]._full_param)
                allign_storage(Embedding)

            self.embeddings.append(Embedding)
        self.embedding = self.embeddings[self.rank]

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore

        input_parallel = copy_to_model_parallel_region(input_)
        # input_parallel = shift_to_model_parallel_region(input_)

        output_list = [None for _ in range(self.world_size)]

        for i in range(self.world_size):
            index = (self.rank +i) % self.world_size

            output = self.embeddings[index](input_parallel)

            output = _WeightParallelRegion_test.apply(output, self.embeddings[index], self.embeddings[(index+1) % self.world_size], i)

            output_list[index] = output

        output_parallel = torch.cat(output_list, dim=-1).contiguous()

        return output_parallel