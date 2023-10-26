data_path=../../data/pretrain
lang=$1
#ulimit -s unlimited
python3 -u preprocessing.py \
        --train_data_file=$data_path/$lang/train.pkl \
        --target_data_file=$data_path/$lang/train.pkl \
        --idiom_loss full \
        --lang=$lang 2>&1| tee full_train_$lang.log
