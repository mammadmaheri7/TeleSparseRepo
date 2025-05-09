#!/bin/bash
#PBS -lwalltime=8:00:00
#PBS -lselect=1:ncpus=32:mem=920gb
#PBS -o Benchmark/frameworks/zkml/logs/output_resnet20.log
#PBS -e Benchmark/frameworks/zkml/logs/error_resnet20.log

# zkml - resnet20 - cifar100 - sparse and teleported

source ~/.config/envman/PATH.env
source ~/.bashrc
source activate base
conda activate zkml_bench_env
cd Benchmark/frameworks/zkml
stdbuf -oL python benchmark.py --size 5 --model resnet20 --save --cores 32  > Benchmark/frameworks/zkml/logs/output_resnet20_live.log 2>&1;
echo "FINISHED ============================== "