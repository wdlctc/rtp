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

from models import transformer_lm_fsdp as transformer_lm
from fairscale.optim import GradScaler

from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union

RPC_PORT = 29501
from data_parallel import ParallelEmbedding, ParallelMultiheadAttention, ColumnParallelLinear
from data_parallel import WeightParallelEmbedding, WeightParallelLinear, WeightParallelMultiheadAttention


rtp_stream = torch.cuda.Stream()

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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

config = {
            "vocab_size": 10000,
            "ninp": 2048,  # embedding dimension
            "nhid": 2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
            "nhead": 32,  # the number of heads in the multiheadattention models
            "dropout": 0,
            "initrange": 0.1,
            "scaler": GradScaler(),
            "clip_value": 0.05,
            "num_decoder_layers": 10,
            "seq_len": 32,
        }

class TestIdenticalOutputs(unittest.TestCase):

    def setUp(self):
        pass


    def assert_grad(self, grad1, grad2):
        assert objects_are_equal(grad1, grad2)

    def test_identical_outputs(self):
        # Example input for the models
        num_samples = 32
        num_embeddings = 32
        embedding_dim = 32
        learning_rate = 0.001
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        init_random_seed(0)

        data = torch.randint(low=0, high=7, size=(num_embeddings, num_samples), dtype=torch.long).cuda()
        labels = torch.randint(low=0, high=7, size=(num_embeddings, num_samples), dtype=torch.long).cuda()

        ninp = config["ninp"]
        nhead = config["nhead"]
        initrange = config["initrange"]
        dropout = config["dropout"]
        vocab_size = config["vocab_size"]
        nhid = config["nhid"]
        ndecoder = config["num_decoder_layers"] - 2
        dropout = 0

        
        sub_embedding_dim = ninp // world_size

        model = transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)
        model.eval()
        Col_model = copy.deepcopy(model)
        Weight_model = copy.deepcopy(model)
        outputs = model(data)
        
        criterion = nn.CrossEntropyLoss().cuda()

        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()

        ref_grads = [p.grad.detach().clone() for p in model.parameters()]
        test_names = [n for n,_ in model.named_parameters()]
        for param in model.parameters():
            param.grad = None
            
        # sub_sample = num_samples // world_size
        # data_list = torch.split(data, sub_sample, dim=1)
        # cur_data = data_list[rank]
        # output_list = torch.split(outputs, sub_sample, dim=1)
        # cur_output = output_list[rank]
        # label_list = torch.split(labels, sub_sample, dim=1)
        # cur_label = label_list[rank].contiguous()
    
        # replace_embedding_layers(Weight_model, WeightParallelEmbedding, world_size, rank)
        # replace_linear_layers(Weight_model, WeightParallelLinear, world_size, rank)
        # replace_attention_layers(Weight_model, WeightParallelMultiheadAttention, world_size, rank)
        # Weight_model.eval()

        # Weight_model_output = Weight_model(cur_data)

        # # print(Weight_model_output - cur_output)
        # assert(torch.max(torch.abs(Weight_model_output - cur_output)) < 1e-4)

        # Weight_loss = criterion(Weight_model_output.view(-1, vocab_size), cur_label.view(-1)) / 2
        # Weight_loss.backward()
        # Weight_grads = []

        # orders = []
        # for i in range(3):
        #     orders += [j * 3 + i for j in range(world_size)]
        # tmp  = []

        # for param_name, param in Weight_model.named_parameters():
        #     tmp.append(param_name)
        #     if 'in_proj' in param_name:
        #         grad = param.grad.clone()
        #         grad = _gather(grad, dim=0)
        #         grad_list = torch.split(grad, sub_embedding_dim, dim=0)
        #         grad_list = [grad_list[i] for i in orders]
        #         grad = torch.cat(grad_list, dim=0).contiguous()
        #         Weight_grads.append(grad)
        #     elif 'out_proj' in param_name and 'weight' in param_name:
        #         grad = param.grad.clone()
        #         grad = _gather(grad, dim=1)
        #         Weight_grads.append(grad)
        #     elif 'out_proj' in param_name and 'bias' in param_name:
        #         grad = param.grad.clone()
        #         Weight_grads.append(grad)
        #     elif 'embedding' in param_name:
        #         grad = param.grad.clone()
        #         grad = _gather(grad, dim=1)
        #         Weight_grads.append(grad)
        #     else:
        #         grad = param.grad.clone()
        #         grad = _gather(grad, dim=0)
        #         Weight_grads.append(grad)

        # for grad1, grad2, name, name2 in zip(ref_grads, Weight_grads, tmp, test_names):
        #     if grad1.shape == grad2.shape:
        #         assert(torch.max(torch.abs(grad1 - grad2)) < 1e-3)

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