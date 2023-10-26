data_path=../../data/pretrain
adv_test_path=../../data/adv_test
key=function
lang=python
python3 -u preprocessing_codesearchnet.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$adv_test_path/test.jsonl \
        --lang=$lang \
        --raw_code_key=$key
