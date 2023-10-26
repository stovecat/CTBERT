#!/bin/bash
LD_PRELOAD=/usr/local/lib/libjemalloc.so 
cuda=$1
lang=python
path=../ckpt/codesearchnet/codeidiombert_only_$lang
init_path=../ckpt/pretrain/codeidiombert_only_python/checkpoint-last
data_path=../data/adv_test
mkdir -p $path
cd ../src
CUDA_VISIBLE_DEVICES=$cuda python inference_codeidiombert_python_adv_test.py \
    --output_dir=$path \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=$init_path \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --train_data_file=$data_path/train.jsonl \
    --eval_data_file=$data_path/valid.jsonl \
    --test_data_file=$data_path/test.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --data_flow_length 0 \
    --idiom_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 1024 \
    --learning_rate 2e-5 \
    --do_test \
    --use_code_idioms \
    --raw_code_key function \
    --not_use_dfg \
    --seed 123456 2>&1| tee $path/adv_test.log
