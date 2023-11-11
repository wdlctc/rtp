import unittest
import argparse
import sys
import copy
# from your_model_file import ModelA, ModelB  # Import your models from your model file

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from fairscale.internal import torch_version

from typing import Any, Optional

RPC_PORT = 29501
from rtp.module.linear import ColumnParallelLinear, RowParallelLinear
from rtp.module.conv import ColumnParallelConv2d
from torch.nn.parallel import DistributedDataParallel as DDP

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

def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = torch.distributed.distributed_c10d._get_default_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def replace_linear_layers(module, world_size, rank):
    """
    Recursively replace all linear layers in the module with the custom layer.
    
    Args:
    - module (nn.Module): The module (or model) to modify.
    - custom_layer_class (nn.Module): The custom layer class to replace with.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None
            device = child.weight.device
            dtype = child.weight.dtype


            if out_features % world_size == 0:
                r = ColumnParallelLinear(in_features, out_features, bias=bias,
                                        linear_layer=child, 
                                        device=device, dtype=dtype)
            else:
                r = RowParallelLinear(in_features, out_features, bias=bias,
                                        linear_layer=child, 
                                        device=device, dtype=dtype)
            setattr(module, name, r)
            
        else:
            replace_linear_layers(child, world_size, rank)

def replace_conv_layers(module, custom_layer_class, world_size, rank):
    """
    Recursively replace all linear layers in the module with the custom layer.
    
    Args:
    - module (nn.Module): The module (or model) to modify.
    - custom_layer_class (nn.Module): The custom layer class to replace with.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size
            stride = child.stride
            padding = child.padding
            dilation = child.dilation
            groups = child.groups
            bias = child.bias is not None
            device = child.weight.device
            dtype = child.weight.dtype

            r = custom_layer_class(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                   Conv2d_layer=child,
                                   device=device, dtype=dtype)
            
            setattr(module, name, r)
        else:
            replace_conv_layers(child, custom_layer_class, world_size, rank)

class TestIdenticalOutputs(unittest.TestCase):

    def setUp(self):
        pass


    def assert_grad(self, grad1, grad2):
        assert objects_are_equal(grad1, grad2)

    def test_identical_outputs(self):
        # Example input for the models
        num_samples = 4
        in_channels = 1
        input_size = 28
        learning_rate = 0.001
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        init_random_seed(0)

        data_list = [torch.randn(num_samples, in_channels, input_size, input_size).cuda() for _ in range(world_size)]
        data =  data_list[rank]#torch.cat(data_list, dim=0).contiguous()
        labels_list = [torch.randint(0, 2, (num_samples,)).cuda() for _ in range(world_size)]
        labels =  labels_list[rank]#torch.cat(labels_list, dim=0).contiguous()

        model = Net().cuda()
        model.eval()

        Weight_model = copy.deepcopy(model)
        Weight_model.eval()

        model = DDP(model)

        # criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

        
        outputs = model(data)
        loss = F.nll_loss(outputs, labels, reduction='sum') #criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()

        ref_grads = [p.grad.detach().clone() for p in model.parameters()]
        # for param in model.parameters():
        #     param.grad = None
            
        optimizer.step()

        sub_sample = num_samples
        # data_list = torch.split(data, sub_sample, dim=0)
        cur_data = data_list[rank]
        #output_list = torch.split(outputs, sub_sample, dim=0)
        cur_output = outputs#output_list[rank]
        # labels_list = torch.split(labels, sub_sample, dim=0)
        cur_label = labels_list[rank]

        replace_linear_layers(Weight_model, world_size, rank)
        replace_conv_layers(Weight_model, ColumnParallelConv2d, world_size, rank)
        Weight_model_output = Weight_model(cur_data)
        
        Weight_optimizer = torch.optim.Adadelta(Weight_model.parameters(), lr=learning_rate)

        assert objects_are_equal(cur_output, Weight_model_output)

        
        Weight_optimizer.zero_grad()
        Weight_loss = F.nll_loss(Weight_model_output, cur_label, reduction='sum') / world_size #criterion(Weight_model_output, cur_label) / world_size
        Weight_loss.backward()
        Weight_grads = [p.grad.detach().clone() for p in Weight_model.parameters()]

        Weight_optimizer.step()
        Weight_optimizer.zero_grad()


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
