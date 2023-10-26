model=$1
path=../ckpt/codesearchnet/$model/predictions.jsonl
data_path=../data/adv_test
python $data_path/evaluator.py -a $data_path/test.jsonl -p $path
