#!/bin/bash
#PBS -lwalltime=8:00:00
#PBS -lselect=1:ncpus=32:mem=920gb
#PBS -o Benchmark/frameworks/ezkl/logs/output_resnet20.log
#PBS -e Benchmark/frameworks/ezkl/logs/error_resnet20.log

# ezkl - resnet20 - cifar100 - sparse and teleported

source ~/.config/envman/PATH.env
source ~/.bashrc
source activate base
conda env list
conda activate zkml_bench_env
cd Benchmark/frameworks/ezkl
stdbuf -oL python benchmark.py --size 1 --model resnet20 --save --cores 32 --sparsity 0 --teleported > Benchmark/frameworks/ezkl/logs/output_resnet20_live.log 2>&1;
echo "FIRST PART (only teleported) FINISHED ============================== "
stdbuf -oL python benchmark.py --size 1 --model resnet20 --save --cores 32 --sparsity 50 > Benchmark/frameworks/ezkl/logs/output_resnet20_live.log 2>&1;
echo "Second PART (only sparse) FINISHED ============================== "
stdbuf -oL python benchmark.py --size 1 --model resnet20 --save --cores 32 --sparsity 50 --teleported > Benchmark/frameworks/ezkl/logs/output_resnet20_live.log 2>&1;
echo "THIRD PART (sparse and teleported) FINISHED ============================== "
stdbuf -oL python benchmark.py --size 1 --model resnet20 --save --cores 32 --sparsity 0 > Benchmark/frameworks/ezkl/logs/output_resnet20_live.log 2>&1;
echo "LAST PART (no-pre-process) FINISHED ============================== "