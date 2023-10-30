import unittest
import argparse
import sys
# from your_model_file import ModelA, ModelB  # Import your models from your model file

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from fairscale.internal import torch_version

from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union

RPC_PORT = 29501
from rtp.module.attention import ParallelMultiheadAttention, WeightParallelMultiheadAttention

def get_ColumnParallelLinear_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    in_features = config["in_features"]
    out_features = config["out_features"]
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    return ColumnParallelLinear(in_features, out_features, world_size=world_size, rank=rank).to(device)


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def objects_are_equal(
    a: Any,
    b: Any,
    raise_exception: bool = False,
    dict_key: Optional[str] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> bool:
    """
    Test that two objects are equal. Tensors are compared to ensure matching
    size, dtype, device and values.
    """
    if type(a) is not type(b):
        if raise_exception:
            raise ValueError(f"type mismatch {type(a)} vs. {type(b)}")
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            if raise_exception:
                raise ValueError(f"keys mismatch {a.keys()} vs. {b.keys()}")
            return False
        for k in a.keys():
            if not objects_are_equal(a[k], b[k], raise_exception, k):
                return False
        return True
    elif isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            if raise_exception:
                raise ValueError(f"length mismatch {len(a)} vs. {len(b)}")
            return False
        return all(objects_are_equal(x, y, raise_exception) for x, y in zip(a, b))
    elif torch.is_tensor(a):
        try:
            # assert_close doesn't strictly test shape, dtype and device
            shape_dtype_device_match = a.size() == b.size() and a.dtype == b.dtype and a.device == b.device
            if not shape_dtype_device_match:
                if raise_exception:
                    msg = f"sizes: {a.size()} vs. {b.size()}, "
                    msg += f"types: {a.dtype} vs. {b.dtype}, "
                    msg += f"device: {a.device} vs. {b.device}"
                    raise AssertionError(msg)
                else:
                    return False
            # assert_close.
            if torch_version() < (1, 12, 0):
                torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
            else:
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            return True
        except (AssertionError, RuntimeError) as e:
            if raise_exception:
                if dict_key and isinstance(e, AssertionError):
                    # Add dict key to the assertion error.
                    msg = e.args[0]
                    new_msg = f"For dict key '{dict_key}': {msg}"
                    raise AssertionError(new_msg) from None
                else:
                    raise e
            else:
                return False
    else:
        return a == b


def _gather(input_: torch.Tensor, dim) -> torch.Tensor:

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output

class TestIdenticalOutputs(unittest.TestCase):

    def setUp(self):
        pass
        

    def test_embedding_identical_outputs(self):
        
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        init_random_seed(0)
        num_samples = 8
        input_size = 8
        embedding_dim = 8
        num_heads = 4
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        sub_embedding_dim = embedding_dim // world_size

        data = torch.randn(num_samples, input_size, embedding_dim).cuda()

        MultiheadAttention = nn.MultiheadAttention(embedding_dim, num_heads, bias=True).cuda()
        MultiheadAttention_output = MultiheadAttention(data, data, data)[0]

        loss = MultiheadAttention_output.sum()
        loss.backward()

        param_names = []
        ref_grads = []
        for param_name, param in MultiheadAttention.named_parameters():
            param_names.append(param_name)
            ref_grads.append(param.grad.clone())

        Model_MultiheadAttention = ParallelMultiheadAttention(embedding_dim, num_heads, bias=True, world_size=world_size, rank=rank, MultiheadAttention_layer=MultiheadAttention).cuda()
        Model_MultiheadAttention_output = Model_MultiheadAttention(data, data, data)[0]

        assert objects_are_equal(MultiheadAttention_output, Model_MultiheadAttention_output)
        
        Model_loss = Model_MultiheadAttention_output.sum()
        Model_loss.backward()
        Model_grads = []

        orders = []
        for i in range(3):
            orders += [j * 3 + i for j in range(world_size)]

        for param_name, param in Model_MultiheadAttention.named_parameters():
            if 'in_proj' in param_name:
                grad = param.grad.clone()
                grad = _gather(grad, dim=0)
                grad_list = torch.split(grad, sub_embedding_dim, dim=0)
                grad_list = [grad_list[i] for i in orders]
                grad = torch.cat(grad_list, dim=0).contiguous()
                Model_grads.append(grad)
            elif 'out_proj' in param_name and 'weight' in param_name:
                grad = param.grad.clone()
                grad = _gather(grad, dim=1)
                Model_grads.append(grad)
            elif 'out_proj' in param_name and 'bias' in param_name:
                grad = param.grad.clone()
                Model_grads.append(grad)

        for grad1, grad2 in zip(ref_grads, Model_grads):
            assert objects_are_equal(grad1, grad2)

        sub_sample = num_samples // world_size
        data_list = torch.split(data, sub_sample, dim=1)
        data = data_list[rank]
        output_list = torch.split(MultiheadAttention_output, sub_sample, dim=1)

        MultiheadAttention_output = MultiheadAttention(data, data, data)[0]
        assert objects_are_equal(output_list[rank], MultiheadAttention_output)

        Weight_MultiheadAttention = WeightParallelMultiheadAttention(embedding_dim, num_heads, bias=True, 
                                                                     world_size=world_size, rank=rank, 
                                                                     device = device, 
                                                                     MultiheadAttention_layer=MultiheadAttention)
        Weight_MultiheadAttention._setup_streams()
        Weight_MultiheadAttention_output = Weight_MultiheadAttention(data, data, data)[0]

        assert objects_are_equal(MultiheadAttention_output, Weight_MultiheadAttention_output)

        Weight_loss = Weight_MultiheadAttention_output.sum()
        Weight_loss.backward()
        Weight_grads = []
        
        for param_name, param in Weight_MultiheadAttention.named_parameters():
            if 'in_proj' in param_name:
                grad = param.grad.clone()
                grad = _gather(grad, dim=0)
                grad_list = torch.split(grad, sub_embedding_dim, dim=0)
                grad_list = [grad_list[i] for i in orders]
                grad = torch.cat(grad_list, dim=0).contiguous()
                Weight_grads.append(grad)
            elif 'out_proj' in param_name and 'weight' in param_name:
                grad = param.grad.clone()
                grad = _gather(grad, dim=1)
                Weight_grads.append(grad)
            elif 'out_proj' in param_name and 'bias' in param_name:
                grad = param.grad.clone()
                Weight_grads.append(grad)

        assert(len(ref_grads) == len(Weight_grads))
        for grad1, grad2 in zip(ref_grads, Weight_grads):
            assert objects_are_equal(grad1, grad2)

        # Activation_MultiheadAttention = ActivationParallelMultiheadAttention(embedding_dim, num_heads, bias=False, world_size=world_size, rank=rank, MultiheadAttention=MultiheadAttention).cuda()
        # Activation_MultiheadAttention_output = Activation_MultiheadAttention(data, data, data)[0]

        # assert objects_are_equal(MultiheadAttention_output, Activation_MultiheadAttention_output)
        # Model_Embedding = ParallelEmbedding(num_embeddings, embedding_dim, world_size=world_size, rank=rank, Embedding_layer=Embedding).cuda()
        # Model_Embedding_output = Model_Embedding(data)

        # assert objects_are_equal(Embedding_output, Model_Embedding_output)

        # Weight_Embedding = WeightParallelEmbedding(num_embeddings, embedding_dim, world_size=world_size, rank=rank, Embedding_layer=Embedding).cuda()
        # Weight_Embedding_output = Weight_Embedding(data)
        
        # assert objects_are_equal(Embedding_output, Weight_Embedding_output)

        # Activation_Embedding = ActivationParallelEmbedding(num_embeddings, embedding_dim, world_size=world_size, rank=rank, Embedding_layer=Embedding).cuda()
        # Activation_Embedding_output = Activation_Embedding(data)

        # assert objects_are_equal(Embedding_output, Activation_Embedding_output)
        # print(Embedding_output.shape, Model_Embedding_output.shape)

        # if rank == 1:
        #     print(Embedding_output[0], Model_Embedding_output[0])


parser = argparse.ArgumentParser()
parser.add_argument('--local-rank', type=int, default=0)
args, remaining = parser.parse_known_args()
sys.argv[1:] = remaining 

# This allows the test to be run if the file is run as a script
if __name__ == '__main__':
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = num_devices
    torch.distributed.init_process_group(
        backend="nccl", rank=args.local_rank, world_size=world_size
    )
    unittest.main()
