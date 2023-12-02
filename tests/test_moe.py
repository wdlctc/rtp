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

from typing import Any, Optional

RPC_PORT = 29501

from rtp.module.moe import MOELayer
from rtp.module.top2gate import Top2Gate
from rtp.module.moe import WeightMOELayer
from rtp.module.top2gate import WeightTop2Gate


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

class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, d_model, dim_feedforward, activation, dropout) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        
class TestIdenticalOutputs(unittest.TestCase):

    def setUp(self):
        pass


    def assert_grad(self, grad1, grad2):
        assert objects_are_equal(grad1, grad2)

    def test_identical_outputs(self):
        # Example input for the models
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        init_random_seed(0)
        num_samples = 4
        input_size = 4
        embedding_dim = 4
        num_heads = 4
        d_model = 4
        dim_feedforward = 4
        activation=nn.ReLU()
        dropout=0
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        num_local_experts = 1
        num_global_experts = num_local_experts * world_size

        sub_embedding_dim = embedding_dim // world_size

        data = torch.randn(num_samples, input_size, embedding_dim).cuda()
        labels = torch.randint(0, 2, (num_samples, input_size)).cuda()

        gate = Top2Gate(d_model, num_global_experts).cuda()
        gate.eval()

        experts_list = [nn.ModuleList(
                [FeedForwardLayer(d_model, dim_feedforward, activation, dropout).cuda() for _ in range(num_local_experts)]
            ) for _ in range(world_size)]
        experts = experts_list[0]
        moe_layer = MOELayer(gate, experts).cuda()
        moe_layer.eval()
        moe_output = moe_layer(data)

        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(moe_output, labels)
        loss.backward()
        ref_grads = [p.grad.detach().clone() for p in moe_layer.parameters()]

        # recheck for this data
        Weight_gate = gate
        
        #WeightTop2Gate(d_model, num_global_experts, gate=gate, device = device).cuda()
        # Weight_experts = nn.ModuleList(
        #         [FeedForwardLayer(d_model, dim_feedforward, activation, dropout).cuda() for _ in range(num_local_experts)]
        #     )
        Weight_moe = WeightMOELayer(Weight_gate, experts, device=device).cuda()
        Weight_moe.eval()
        Weight_moe_output = Weight_moe(data)

        # if rank == 0:
        #     print('---------------------------------')
        #     print((moe_output - Weight_moe_output))
        assert(torch.max(torch.abs(moe_output - Weight_moe_output)) < 1e-5)

        Weight_loss = criterion(Weight_moe_output, labels).div_(world_size)
        Weight_loss.backward()

        Weight_grads = [p.grad.detach().clone() for p in Weight_moe.parameters()]


        assert(len(ref_grads) == len(Weight_grads))
        
        count = 0
        # for grad1, grad2 in zip(ref_grads, Weight_grads):
        #     if rank == 0:
        #         print(torch.max(torch.abs(grad1 - grad2)))
            # assert(torch.max(torch.abs(grad1 - grad2))< 1e-6)



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
