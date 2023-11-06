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
from .collectives import gather_from_model_parallel_region, copy_to_model_parallel_region
from .collectives import set_full_param, allign_storage, free_storage, set_full_param2
from .collectives import _WeightParallelRegion_all, _WeightParallelRegion_test

from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn
from .utils import divide_and_check_no_remainder, divide, split_tensor_along_last_dim

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
from .collectives import set_full_param, set_full_param2, allign_storage, free_storage, _WeightParallelRegion_all, _WeightParallelRegion_test
from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn




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


from typing import Callable, Optional, Tuple, Any, List, Union
import math
import torch
import torch.nn as nn
from .functional import multi_head_attention_forward
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch import Tensor
from functools import partial

from .utils import divide_and_check_no_remainder, affine_weight, affine_weight_attention
from .collectives import gather_from_model_parallel_region, reduce_from_model_parallel_region, shift_to_model_parallel_region, copy_to_model_parallel_region
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from .collectives import set_full_param, set_full_param2, allign_storage, free_storage, _WeightParallelRegion_test, _WeightParallelRegion_attention
from .collectives import _WeightParallelRegion_before, _WeightParallelRegion_after, hook_fn

class SubParallelMultiheadAttention(torch.nn.Module):
    
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, world_size, rank, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, MultiheadAttention=None, empty_init=False) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SubParallelMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.world_size = world_size
        self.rank = rank
        self.num_heads_per_partition = divide_and_check_no_remainder(self.num_heads, self.world_size)
        self.embed_dim_per_partition = divide_and_check_no_remainder(self.embed_dim, self.world_size)

        embed_dim_per_partition = self.embed_dim_per_partition

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim_per_partition, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim_per_partition, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim_per_partition, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim_per_partition, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        self.bias = bias
        if bias:
            self.in_proj_bias = Parameter(torch.empty((3 * embed_dim_per_partition), **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim_per_partition, embed_dim, bias=bias, **factory_kwargs)\

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def affine_weight(self, MultiheadAttention):

        if MultiheadAttention is not None:
            if not self._qkv_same_embed_dim:
                affine_weight(self.q_proj_weight, MultiheadAttention.q_proj_weight, self.embed_dim_per_partition, 0, self.world_size, self.rank)
                affine_weight(self.k_proj_weight, MultiheadAttention.k_proj_weight, self.embed_dim_per_partition, 0, self.world_size, self.rank)
                affine_weight(self.v_proj_weight, MultiheadAttention.v_proj_weight, self.embed_dim_per_partition, 0, self.world_size, self.rank)
            else:
                affine_weight_attention(self.in_proj_weight, 
                                        MultiheadAttention.in_proj_weight, 
                                        [self.rank, self.rank+self.world_size, self.rank+self.world_size*2],
                                        self.embed_dim_per_partition, 
                                        0, 
                                        self.world_size, 
                                        self.rank)
            if self.bias:
                affine_weight_attention(self.in_proj_bias, 
                                        MultiheadAttention.in_proj_bias, 
                                        [self.rank, self.rank+self.world_size, self.rank+self.world_size*2],
                                        self.embed_dim_per_partition, 
                                        0, 
                                        self.world_size, 
                                        self.rank)
            affine_weight(self.out_proj.weight, MultiheadAttention.out_proj.weight, self.embed_dim_per_partition, 1, self.world_size, self.rank)
            if self.bias:
                self.out_proj.bias.data.copy_(MultiheadAttention.out_proj.bias.data)
                self.out_proj.bias.data.div_(self.world_size)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        query_parallel = query
        key_parallel = key
        value_parallel = value

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask, 
            mask_name='key_padding_mask',
            other_type=F._none_or_dtype(attn_mask),
            other_name='attn_mask',
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask, 
            mask_name='attn_mask',
            other_type=None,
            other_name="",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim_per_partition, self.num_heads_per_partition,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                E_div=self.world_size)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim_per_partition, self.num_heads_per_partition,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                E_div=self.world_size)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class ParallelMultiheadAttention(torch.nn.Module):
    
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, world_size, rank, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, MultiheadAttention_layer=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.world_size = world_size
        self.rank = rank
        self.num_heads_per_partition = divide_and_check_no_remainder(self.num_heads, self.world_size)
        self.embed_dim_per_partition = divide_and_check_no_remainder(self.embed_dim, self.world_size)

        self.MultiheadAttention = SubParallelMultiheadAttention(embed_dim,
                                                                num_heads,
                                                                world_size,
                                                                rank,
                                                                dropout,
                                                                bias,
                                                                add_bias_kv,
                                                                add_zero_attn,
                                                                kdim,
                                                                vdim,
                                                                batch_first,
                                                                device,
                                                                dtype)

        self.MultiheadAttention.affine_weight(MultiheadAttention_layer)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:

        query = copy_to_model_parallel_region(query)
        key = copy_to_model_parallel_region(query)
        value = copy_to_model_parallel_region(query)
        
        attn_output, attn_output_weights = self.MultiheadAttention(query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)

        attn_output = reduce_from_model_parallel_region(attn_output)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

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
        world_size: int = 1,
        rank: int = 0,
        device=None,
        dtype=None,
        gather_output = True,
        Conv2d_layer = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ColumnParallelConv2d, self).__init__()

        # Keep input parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(out_channels, world_size)
        self.rank = rank
        self.world_size = world_size

        self.conv = ModelParallelConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **factory_kwargs,
                                        world_size = world_size, rank = rank)
        
        self.conv.affine_weight(Conv2d_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        
        output_parallel = self.conv(input_parallel)
        if self.gather_output:
            output = gather_from_conv_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output
