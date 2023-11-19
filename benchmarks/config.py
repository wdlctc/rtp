import argparse
import os
from pathlib import Path
import json
import torch.nn as nn

from fairscale.optim import GradScaler

def parse_args():
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--max_batch", type=int, default=4, help="Max number of batches")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
    parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
    parser.add_argument(
        "--model_name",
        default="lm",
        help="Language Model(LM) used to benchmark FSDP.",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Display additional debug information")
    parser.add_argument("--enable_auto_wrap", action="store_true", default=False, help="Use auto_wrap with FSDP")
    parser.add_argument("--benchmark_eval", action="store_true", default=False, help="Benchmark evaluation workflow.")
    parser.add_argument("--full_fp16", action="store_true", default=False, help="Benchmark in full fp16 mode.")
    # Model Config arguments
    parser.add_argument("--vocab_size", type=int, default=50256)
    parser.add_argument("--ninp", type=int, default=1280)
    parser.add_argument("--nhid", type=int, default=5120)
    parser.add_argument("--nhead", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--initrange", type=float, default=0.1)
    parser.add_argument("--clip_value", type=float, default=0.05)
    parser.add_argument("--num_decoder_layers", type=int, default=36)
    parser.add_argument("--seq_len", type=int, default=1025)
    
    # Benchmark Config arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # load from model name
    parser.add_argument("--model_config", type=str, default='None')
    return parser.parse_args()

MODEL_CONFIG_DIR_NAME = "./model_configs"

def read_configs(config_dir_name: str) -> dict:
    """Read configs from a directory."""
    configs = {}
    for filename in os.listdir(config_dir_name):
        filepath = os.path.join(config_dir_name, filename)
        with open(filepath, "r") as f:
            config_json = json.load(f)
            configs[config_json['name']] = config_json
    return configs

# def get_model_config_by_name(name_or_path: str) -> ModelConfig:
#     """Get model config from the populated mapping by name, or from model config json file path, if not found from the previous methods, try to get it from HuggingFace."""
#     if name_or_path in model_configs:
#         return model_configs[name_or_path]
#     if os.path.isfile(name_or_path) and ".json" in name_or_path:
#         try:
#             with open(name_or_path, "r") as f:
#                 config_json = json.load(f)
#                 config = ModelConfig(**config_json)
#                 if config.name not in model_configs:
#                     model_configs[config.name] = config
#             return config
#         except Exception as e:
#             raise ValueError(f"unknown gpu config name: {e}")
#     model_config = get_model_config_from_hf(name_or_path)
#     if model_config is None:
#         raise (
#             f"unknown model config name: {name_or_path}, and none is found on HuggingFace Hub"
#         )
#     return model_config

class FSDP:
    def get_model_config(args):
        default_config = {
            "vocab_size": args.vocab_size,
            "ninp": args.ninp,
            "nhid": args.nhid,
            "nhead": args.nhead,
            "dropout": args.dropout,
            "initrange": args.initrange,
            "scaler": GradScaler(),
            "clip_value": args.clip_value,
            "num_decoder_layers": args.num_decoder_layers,
            "seq_len": args.seq_len,
        }
        model_configs = read_configs(os.path.join(Path(__file__).parent, MODEL_CONFIG_DIR_NAME))
        if args.model_config in model_configs:
            model_config_json = model_configs[args.model_config]
            default_config['num_decoder_layers'] = model_config_json["num_layers"]
            default_config['nhead'] = model_config_json["n_head"]
            default_config['ninp'] = model_config_json["hidden_dim"]
            if model_config_json["vocab_size"] % default_config['num_decoder_layers'] != 0:
                model_config_json["vocab_size"] = model_config_json["vocab_size"] // model_config_json["n_head"] * model_config_json["n_head"]
            default_config['vocab_size'] = model_config_json["vocab_size"]
            default_config['seq_len'] = model_config_json["max_seq_len"] + 1
            default_config['nhid'] = model_config_json["ffn_embed_dim"]
            
        return default_config

    def get_benchmark_config(args):

        return {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "criterion": nn.CrossEntropyLoss(),
        }

    def get_golden_real_stats():
        raise NotImplementedError("Synthetic data benchmarks are not supported.")

    def get_golden_synthetic_stats():
        return {
            "avg_wps": 486.303,
            "std_dev_wps": 71.307,
            "peak_mem_usage": [5.5055 * 2**30, 5.5055 * 2**30, 5.5055 * 2**30, 5.5055 * 2**30],
        }
