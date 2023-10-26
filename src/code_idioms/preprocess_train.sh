data_path=../../data/pretrain
lang=$1
python3 -u preprocessing.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$data_path/$lang/train.pkl \
        --lang=$lang 2>&1| tee train_$lang.log
