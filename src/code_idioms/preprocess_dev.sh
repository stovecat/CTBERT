data_path=../../data/pretrain
lang=$1
python3 -u preprocessing.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$data_path/$lang/dev.pkl \
        --lang=$lang 2>&1| tee full_dev_$lang.log
