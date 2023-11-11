import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from rtp.rotated_tensor_parallel import RotatedTensorParallel
import numpy as np

from typing import Any, Optional
from fairscale.internal import torch_version

RPC_PORT = 29503

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

def split_tensor(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, dim: int = -1
):
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    dim_size = tensor.size()[dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def benchmark_fsdp(rank, world_size, args):
    """Benchmark a given model using a single process and multiple devices."""
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )
    torch.cuda.set_device(rank)
    init_random_seed(0)
    model = Net()
    ref = copy.deepcopy(model)

    model = RotatedTensorParallel(model)
    model.cuda()

    model.eval()
    example = torch.randn(1, 1, 28, 28).cuda()
    labels = torch.tensor([1]).cuda()

    output = model(example)

    ref.cuda()
    ref.eval()
    output2 = ref(example)
    assert objects_are_equal(output, output2)
    
    loss = F.nll_loss(output2, labels, reduction='sum') #criterion(outputs, labels)
    loss.backward()

    loss = F.nll_loss(output, labels, reduction='sum').div_(world_size) #criterion(outputs, labels)
    loss.backward()

    for param1, param2 in zip(model.parameters(), ref.parameters()):
        grad1 = param1.grad
        grad2 = param2.grad
        if rank == 0:
            if grad1.shape[0] * world_size == grad2.shape[0]:
                grad2 = split_tensor(grad2, num_partitions=world_size, dim=0)[rank]
                # print(grad1.shape, grad2.shape)
                # print(torch.max(torch.abs(grad1 - grad2)))
                # print(grad1, grad2)
                assert objects_are_equal(grad1, grad2)
        # print(grad1.shape, grad2.shape)
        # assert objects_are_equal(param1.grad, param2.grad)

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

if __name__ == '__main__':
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())
    mp.spawn(
        benchmark_fsdp,
        args=(num_devices, args),
        nprocs=num_devices,
        join=True,
    )