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
        
        super(ParallelMultiheadAttention, self).__init__()

        self.world_size = world_size
        self.rank = rank

        self.layers = []
        for i in range(self.world_size):
            MultiheadAttention = SubParallelMultiheadAttention(embed_dim, 
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
                                                                dtype,)

            if i == 0:
                set_full_param(MultiheadAttention, device, dtype)
                allign_storage(MultiheadAttention)
                MultiheadAttention.affine_weight(MultiheadAttention_layer)
            else:
                set_full_param2(MultiheadAttention, device, dtype, self.layers[0]._full_param)
                allign_storage(MultiheadAttention)

            self.layers.append(MultiheadAttention)
        self.MultiheadAttention = self.layers[self.rank]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False,
        test_tensor = None
    ) -> Tuple[Tensor, Optional[Tensor]]:

        for i in range(self.world_size):
            index = (self.rank + i) % self.world_size

            query = _WeightParallelRegion_before.apply(query, self.layers[index], i, self)

            output, attn_output_weights = self.layers[index](query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal)

            output = _WeightParallelRegion_after.apply(output, self.layers[index], self.layers[(index+1) % self.world_size], i, self)

            if i == 0:
                output_parallel = output
            else:
                output_parallel = output + output_parallel
            

        for i, layer in enumerate(self.layers):
            if i == self.rank:
                continue
            layer.count = 0
            for p in layer.parameters():
                p_tmp = p.expand_as(p)
                assert p_tmp.grad_fn is not None
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                layer.count += 1
                handle = grad_acc.register_hook(partial(hook_fn, p, layer, self))
                

        return output_parallel, attn_output_weights
