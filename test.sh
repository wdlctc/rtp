python benchmarks/rtp_benchmark.py --use_synthetic_data --model_config=gpt2-large --batch_size=6
python benchmarks/fsdp_benchmark.py --use_synthetic_data --model_config=gpt2-large --batch_size=6
python benchmarks/dp_benchmark.py --use_synthetic_data --model_config=gpt2-large --batch_size=6


python benchmarks/rtp_benchmark.py --use_synthetic_data --model_config=EleutherAI_gpt-neo-1.3B
python benchmarks/fsdp_benchmark.py --use_synthetic_data --model_config=EleutherAI_gpt-neo-1.3B
python benchmarks/dp_benchmark.py --use_synthetic_data --model_config=EleutherAI_gpt-neo-1.3B


python benchmarks/fsdp_benchmark.py --use_synthetic_data --model_config=EleutherAI_gpt-neo-1.3B