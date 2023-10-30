import argparse
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
    parser.add_argument("--vocab_size", type=int, default=10240)
    parser.add_argument("--ninp", type=int, default=2048)
    parser.add_argument("--nhid", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--initrange", type=float, default=0.1)
    parser.add_argument("--clip_value", type=float, default=0.05)
    parser.add_argument("--num_decoder_layers", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=128)
    
    # Benchmark Config arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

class FSDP:
    def get_model_config(args):
        return {
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