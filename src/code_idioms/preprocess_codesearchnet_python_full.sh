data_path=../../data/pretrain
codesearchnet_path=../../data/codesearchnet
key=original_string
lang=$1
python3 -u preprocessing_codesearchnet.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$codesearchnet_path/$lang/train.jsonl \
        --lang=$lang \
        --idiom_loss full \
        --raw_code_key=$key
python3 -u preprocessing_codesearchnet.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$codesearchnet_path/$lang/codebase.jsonl \
        --lang=$lang\
        --idiom_loss full \
        --raw_code_key=$key
