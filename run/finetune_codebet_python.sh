LD_PRELOAD=/usr/local/lib/libjemalloc.so 
lang=$1
cuda=$2
path=../ckpt/codesearchnet/codebert_python_$lang
init_path=../ckpt/pretrain/codebert_python/checkpoint-last
data_path=../data/codesearchnet
mkdir -p $path
cd ../src
CUDA_VISIBLE_DEVICES=$cuda python finetune_codeidiombert_python.py \
    --output_dir=$path \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=$init_path \
    --tokenizer_name=microsoft/codebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=$data_path/$lang/train.jsonl \
    --eval_data_file=$data_path/$lang/valid.jsonl \
    --test_data_file=$data_path/$lang/test.jsonl \
    --codebase_file=$data_path/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --data_flow_length 0 \
    --idiom_length 0 \
    --train_batch_size 32 \
    --eval_batch_size 256 \
    --learning_rate 2e-5 \
    --do_test \
    --not_use_dfg \
    --seed 123456 2>&1| tee $path/train.log