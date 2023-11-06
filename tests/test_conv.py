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
from rtp.module.conv import ColumnParallelConv2d

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


    def assert_grad(self, grad1, grad2):
        assert objects_are_equal(grad1, grad2)

    def test_identical_outputs(self):
        # Example input for the models
        num_samples = 8
        input_size = 8
        in_channels = 8
        out_channels = 8
        kernel_size = 3
        padding = 1
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        init_random_seed(0)

        data = torch.randn(num_samples, in_channels, input_size, input_size).cuda()

        Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding).cuda()
        Conv_output = Conv(data)

        loss = Conv_output.sum()
        loss.backward()
        ref_grads = [p.grad.detach().clone() for p in Conv.parameters()]

        sub_sample = num_samples // world_size
        data_list = torch.split(data, sub_sample, dim=0)
        cur_data = data_list[rank]
        output_list = torch.split(Conv_output, sub_sample, dim=0)
        cur_output = output_list[rank]

        Weight_conv = ColumnParallelConv2d(in_channels, out_channels,
                                           kernel_size=kernel_size, padding=padding,
                                           Conv2d_layer = Conv)
        # Weight_conv.cuda()
        Weight_conv.to(device)
        Weight_conv_output = Weight_conv(cur_data)

        assert objects_are_equal(cur_output, Weight_conv_output)

        Weight_loss = Weight_conv_output.sum()
        torch.distributed.all_reduce(Weight_loss, op=torch.distributed.ReduceOp.SUM)
        Weight_loss.backward()

        Weight_grads = [p.grad.detach().clone() for p in Weight_conv.parameters()]
        
        assert(len(ref_grads) == len(Weight_grads))
        for grad1, grad2 in zip(ref_grads, Weight_grads):
            grad = _gather(grad2, dim=0)
            assert objects_are_equal(grad, grad1)


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
