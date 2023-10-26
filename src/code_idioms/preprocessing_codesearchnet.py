from preprocessing import *

""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The data file for idiom mining")
    parser.add_argument("--target_data_file", default=None, type=str,
                        help="The target data file to extract code idioms")
    parser.add_argument("--lang", default=None, type=str,
                        help="language")
    parser.add_argument('--data_path', type=str, default='data',
                        help="path for code idioms")
    parser.add_argument('--idiom_path', type=str, default='',
                        help="path for code idioms")
    parser.add_argument('--ptsg_threshold', type=float, default=0.2,
                        help="pTSG threshold for tree fragments")
    parser.add_argument('--depth_threshold', type=int, default=2,
                        help="Depth threshold for tree fragments")
    parser.add_argument('--num_of_nodes_threshold', type=int, default=4,
                        help="Num. of nodes threshold for tree fragments")
    parser.add_argument('--frequency_threshold', type=float, default=0.00,
                        help="Frequency threshold for tree fragments")
    parser.add_argument('--top_k', type=int, default=100,
                        help="Top k idioms")
    parser.add_argument('--corpus_size', type=float, default=0,
                        help="Size of training dataset")
    parser.add_argument('--raw_code_key', type=str, default='original_string',
                        help="Raw code key of the data dictionaries")
    parser.add_argument('--idiom_loss', type=str, default='span',
                        help="Idiom loss type")    
    
    def get_tokenizer(model_name='microsoft/graphcodebert-base'):
        tokenizer_name = model_name
        config = RobertaConfig.from_pretrained(model_name)
        return RobertaTokenizer.from_pretrained(tokenizer_name)
    
    def pack_data(data, tokenizer, args):
        args.corpus_size = len(data)
        items=[]
        for d in data:
            items.append((d,tokenizer,args))
        return items
    
    cpu_cont = 16
    tree_dict = None
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()
    
    #Generate folders if not exist
    idiom_path = get_idiom_path(args, "".join(args.target_data_file.split('/')[-1].split('.')[:-1]), False)
    data_path = get_idiom_path(args, 'ast', True)
    args.idiom_path = './'+args.data_path+'/'+"/".join(idiom_path.split('/')[:-1])
    args.data_path = './'+args.data_path+'/'+data_path+'/ast'

    from pathlib import Path
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    Path(args.idiom_path).mkdir(parents=True, exist_ok=True)
    
    tokenizer = get_tokenizer()
    target_data = load_jsonl(args.target_data_file)
    items = pack_data(target_data, tokenizer, args)
#     print('Mining Code Idioms and Apply to the Training dataset.',flush=True)
    print('Code idiom mining for ',args.lang,'language', flush=True)
        
    code_idioms = load_code_idioms(args, pool, items, tokenizer)
    for idx, v in enumerate(code_idioms[:3]):
        print('Index', str(idx)+':', v, '\n')

