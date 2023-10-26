# coding=utf-8
from finetune_codeidiombert_python import *

# +
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}
logger = logging.getLogger(__name__)
cpu_cont = 16

_parser = {}
for lang in dfg_function.keys():
    LANGUAGE = Language('parser/my-languages.so', lang)
    p = Parser()
    p.set_language(LANGUAGE) 
    _parser[lang] = [p,dfg_function[lang]]


# -

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,
                 idx,
                 idiom_to_code=None
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.idiom_to_code=idiom_to_code
        self.idx=idx


def convert_examples_to_features(item):
    #parsing
    args = item[2]
    if args.use_code_idioms:
        js,tokenizer,args,lang,ci=item
        ci = list(ci)
    else:
        js,tokenizer,args,lang=item
        ci = []
    global _parser
    parser = _parser[lang]
    code_tokens,dfg=extract_dataflow(js[args.raw_code_key],parser,lang)
    cig=ci
    if args.not_use_dfg:
        dfg = []
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]

    #truncating
    code_tokens=code_tokens[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    if args.use_code_idioms:
        position_idx = [i+tokenizer.pad_token_id + 2 for i in range(len(source_tokens))]
        _truncate_seq_pair(dfg,cig, max(args.block_size-len(source_tokens)-1, 0))
    else:
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:max(0,args.block_size-len(source_tokens))]
    source_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
    if args.use_code_idioms and len(source_tokens) < args.block_size:
        idiom_tokens=[tokenizer.sep_token]+[get_idiom_token(x[0], args.lang) for x in cig]
        source_tokens+=idiom_tokens
        position_idx+=[0]+[2 for x in cig]
        source_ids+=tokenizer.convert_tokens_to_ids(idiom_tokens)
    
    padding_length=args.block_size-len(source_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    source_ids+=[tokenizer.pad_token_id]*padding_length

    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
#     length=len([tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token])
#     dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    dfg_to_code=[(x[0]+1,x[1]+1) for x in dfg_to_code]

    idiom_to_code = None
    if args.use_code_idioms:
        def get_span(cig, ori2cur_pos, args, tokenizer):
            idiom_to_code = []
            for x in cig:
                _id = x[0]
                if x[1][0] in ori2cur_pos.keys():
                    _start = ori2cur_pos[x[1][0]][0]
                else:
                    # out of the scope
                    _start = 9999
                if x[1][1] in ori2cur_pos.keys():
                    _end = ori2cur_pos[x[1][1]][1]
                else:
                    # truncated span
                    _end = ori2cur_pos[max(list(ori2cur_pos.keys()))][1]
                idiom_to_code.append((_id, (_start,_end)))
            return [(x[0], (x[1][0]+1,x[1][1]+1)) for x in idiom_to_code]

        def get_full(cig, ori2cur_pos, args, tokenizer):
            idiom_to_code = []
            for x in cig:
                _id = x[0]
                start_end = []
                for _x in x[1]:
                    if _x[0] in ori2cur_pos.keys():
                        _start = ori2cur_pos[_x[0]][0]
                    else:
                        # out of the scope
                        _start = 9999
                    if _x[1] in ori2cur_pos.keys():
                        _end = ori2cur_pos[_x[1]][1]
                    else:
                        # truncated span
                        _end = ori2cur_pos[max(list(ori2cur_pos.keys()))][1]
                    start_end.append((_start+1,_end+1))
                idiom_to_code.append((_id, start_end))
            return idiom_to_code
        try:
            if args.idiom_loss == 'span':
                idiom_to_code = get_span(cig, ori2cur_pos, args, tokenizer)
            elif args.idiom_loss in ['full', 'none']:
                idiom_to_code = get_full(cig, ori2cur_pos, args, tokenizer)
            else:
                raise NotImplementedError
        except KeyError:
            error_dir = 'errors'
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            dump_pkl(error_dir+'/key_error.pkl', (ori2cur_pos, cig, source_tokens, code_tokens, js))
            print('\ncig\n',cig)
            print(x[1][0] in ori2cur_pos.keys(), x[1][0] in ori2cur_pos.keys())
            print(x[1][0] in ori2cur_pos.keys(), x[1][1] in ori2cur_pos.keys())
            assert 1 == 2
            
    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length
    
    return InputFeatures(source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'],js['idx'],idiom_to_code)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None, idiom_type=None):
        self.args=args
        prefix='adv_test'
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        if os.path.exists(cache_file):
            self.examples=load_pkl(cache_file)
        else:
            self.examples = []
            if args.use_code_idioms and idiom_type is not None:
                idiom_path = get_idiom_path(args, idiom_type, run=True)                
                ci_list = load_pkl(idiom_path)
                data=[]
                with open(file_path, 'r', encoding='utf-8') as f:
#                     assert sum([1 for _ in f]) == len(ci_list)
                    for idx, (line, ci) in enumerate(zip(f, ci_list)):
                        line=line.strip()
                        js=json.loads(line)
                        if args.use_code_idioms:
                            if idiom_type is not None:
                                # train / codebase
                                data.append((js,tokenizer,args,args.lang,ci))
                            else:
                                # test nls
                                data.append((js,tokenizer,args,args.lang,[]))
                        else:
                            data.append((js,tokenizer,args,args.lang))
                    assert len(data) == len(ci_list)
            else:
                data=[]
                with open(file_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        line=line.strip()
                        js=json.loads(line)
                        if args.use_code_idioms:
                            data.append((js,tokenizer,args,args.lang,[]))
                        else:
                            data.append((js,tokenizer,args,args.lang))
            r = pool.map_async(convert_examples_to_features, tqdm(data,total=len(data)))
            r.wait()
            self.examples = r.get()
#             for d in tqdm(data):
#                 self.examples.append(convert_examples_to_features(d))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))  
                if args.use_code_idioms:
                    logger.info("idiom_to_code: {}".format(' '.join(map(str, example.idiom_to_code))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.block_size,
                            self.args.block_size),dtype=np.bool)
        #calculate begin index of node and max length of input
        if self.args.use_code_idioms:
            node_index=sum([i>2 for i in self.examples[item].position_idx])
        else:
            node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
        if self.args.use_code_idioms:
            dfg_index = len(self.examples[item].dfg_to_code)+1
            for idx,(ci,v) in enumerate(self.examples[item].idiom_to_code):
                # Option 1
                if self.args.idiom_loss == 'span':
                    a, b = v
                    if a < max_length and b-1 < max_length and a != b-1:
                        attn_mask[idx+node_index+dfg_index,a]=True
                        attn_mask[idx+node_index+dfg_index,b-1]=True

                        attn_mask[a,idx+node_index+dfg_index]=True
                        attn_mask[b-1,idx+node_index+dfg_index]=True
                # Option 2
                elif self.args.idiom_loss in ['full', 'none']:
                    for a, b in v:
                        # Option 2
                        if a < max_length and b-1 < max_length and a != b-1:
                            attn_mask[idx+node_index+dfg_index,a:b]=True
                            attn_mask[a:b,idx+node_index+dfg_index]=True
                else:
                    raise NotImplementedError
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids))


# +
# class TextDataset(Dataset):
#     def __init__(self, tokenizer, args, file_path=None,pool=None, idiom_type=None):
#         self.args=args
#         prefix='adv_test'
#         cache_file=args.output_dir+'/'+prefix+'.pkl'
        
#         if os.path.exists(cache_file):
#             self.examples=pickle.load(open(cache_file,'rb'))
#         else:

#             self.examples = []
#             if args.use_code_idioms:
#                 def get_idiom_path():                    
#                     idiom_option = 'T:'+str(args.ptsg_threshold)+\
#                                    '_F:'+str(args.frequency_threshold)+\
#                                    '_D:'+str(args.depth_threshold)+\
#                                    '_N:'+str(args.num_of_nodes_threshold)+\
#                                    '_K:'+str(args.top_k)
#                     return args.idiom_path+\
#                            '/'.join([idiom_option, \
#                                      args.train_data_file.split('/')[0], \
#                                      ".".join(args.train_data_file.split('/')[1].split('.')[:-1]), \
#                                      'code_idioms.pkl'])
#                 print(args.train_data_file, flush=True)
#                 if 'code_idioms.pkl' not in args.idiom_path:
#                     args.idiom_path = get_idiom_path()
#                 print(args.idiom_path, flush=True)
#                 assert 1 == 2
#                 data=[]
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     for idx, (line, ci) in enumerate(zip(f, pickle.load(open(args.idiom_path,'rb')))):
#                         line=line.strip()
#                         js=json.loads(line)
#                         if args.use_code_idioms:
#                             if idiom_type is not None:
#                                 # train / codebase
#                                 data.append((js,tokenizer,args,args.lang,ci))
#                             else:
#                                 # test nls
#                                 data.append((js,tokenizer,args,args.lang,[]))
#                         else:
#                             data.append((js,tokenizer,args,args.lang))
#             else:
#                 data=[]
#                 for idx,js in enumerate(pickle.load(open(filename,'rb'))):
#                     if len(js['docstring_tokens'])!=0:
#                         data.append((js,tokenizer,args,args.lang))
#             r = pool.map_async(convert_examples_to_features, tqdm(data,total=len(data)))
#             r.wait()
#             self.examples = r.get()
# #             for d in tqdm(data):
# #                 self.examples.append(convert_examples_to_features(d))
#             pickle.dump(self.examples,open(cache_file,'wb'))
            
#         for idx, example in enumerate(self.examples[:3]):
#             logger.info("*** Example ***")
#             logger.info("idx: {}".format(idx))
#             logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
#             logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
#             logger.info("position_idx: {}".format(example.position_idx))
#             logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
#             logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
#             logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
#             logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))  
#             if args.use_code_idioms:
#                 logger.info("idiom_to_code: {}".format(' '.join(map(str, example.idiom_to_code))))

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):
#         #calculate graph-guided masked function
#         attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length+self.args.idiom_length,
#                             self.args.code_length+self.args.data_flow_length+self.args.idiom_length),dtype=np.bool)
#         #calculate begin index of node and max length of input
#         node_index=sum([i>2 for i in self.examples[item].position_idx])
#         max_length=sum([i!=1 for i in self.examples[item].position_idx])
#         #sequence can attend to sequence
#         attn_mask[:node_index,:node_index]=True
#         #special tokens attend to all tokens
#         for idx,i in enumerate(self.examples[item].code_ids):
#             if i in [0,2]:
#                 attn_mask[idx,:max_length]=True
#         #nodes attend to code tokens that are identified from
#         for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
#             if a<node_index and b<node_index:
#                 attn_mask[idx+node_index,a:b]=True
#                 attn_mask[a:b,idx+node_index]=True
#         #nodes attend to adjacent nodes 
#         for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
#             for a in nodes:
#                 if a+node_index<len(self.examples[item].position_idx):
#                     attn_mask[idx+node_index,a+node_index]=True  
#         if self.args.use_code_idioms:
#             dfg_index = len(self.examples[item].dfg_to_code)+1
#             for idx,(ci,(a,b)) in enumerate(self.examples[item].idiom_to_code):
#                 # Option 1
#                 if a < max_length and b-1 < max_length and a != b-1:
#                     attn_mask[idx+node_index+dfg_index,a]=True
#                     attn_mask[idx+node_index+dfg_index,b-1]=True

#                     attn_mask[a,idx+node_index+dfg_index]=True
#                     attn_mask[b-1,idx+node_index+dfg_index]=True
#                 # Option 2
# #                     self.attn_mask[item,idx+node_index+dfg_index,a:b]=True
# #                     self.attn_mask[item,a:b,idx+node_index+dfg_index]=True
#                 if torch.tensor(attn_mask[idx+node_index+dfg_index,:]).int().sum() == 0:
#                     #skipped due to truncate
#                     pass
#         return (torch.tensor(self.examples[item].code_ids),
#               torch.tensor(attn_mask),
#               torch.tensor(self.examples[item].position_idx), 
#               torch.tensor(self.examples[item].nl_ids))
# -

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# +
def test(args, model, tokenizer, pool):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file, pool=pool, idiom_type='test')


#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)# if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    code_vecs=[] 
    nl_vecs=[]
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)  
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)

        with torch.no_grad():
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            eval_loss += lm_loss.mean().item()
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores=np.matmul(nl_vecs,code_vecs.T)

    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    indexs=[]
    urls=[]
    for example in eval_dataset.examples:
        indexs.append(example.idx)
        urls.append(example.url)
    with open(os.path.join(args.output_dir,"predictions.jsonl"),'w') as f:
        for index,url,sort_id in zip(indexs,urls,sort_ids):
            js={}
            js['url']=url
            js['answers']=[]
            for idx in sort_id[:100]:
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js)+'\n')


# +
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    ###############################################################################################################
    # Code idioms
    ###############################################################################################################
    parser.add_argument('--use_code_idioms', action='store_true',
                        help="Use code idioms")
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
    parser.add_argument('--idiom_path', type=str, default='./code_idioms/data/',
                        help="Idiom path")
    parser.add_argument("--idiom_length", default=192, type=int,
                        help="Optional Idiom input sequence length after tokenization.")     
    parser.add_argument('--raw_code_key', type=str, default='original_string',
                        help="Raw code key of the data dictionaries")
    parser.add_argument('--not_use_dfg', action='store_true',
                        help="disable data flow")    
    parser.add_argument('--idiom_loss', type=str, default='span',
                        help="Idiom loss type")    
    ###############################################################################################################
    ###############################################################################################################

    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()

    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    
    # Code idioms
    if args.use_code_idioms:
        # for the position id of code idioms (2)
        def expand_position_embeddings(pretrained):
            config.max_position_embeddings = 515
            _model = RobertaModel(config)
            placeholder = _model.state_dict()
            state_dict={k:v if v.size()==placeholder[k].size() else placeholder[k] for k,v in zip(placeholder.keys(), pretrained.state_dict().values())}
            _model.load_state_dict(state_dict)
            return _model
        
        if args.use_code_idioms and (model.embeddings.position_embeddings.num_embeddings < 515):
            model = expand_position_embeddings(model)
            print('Position embeddings are expanded:',model.embeddings.position_embeddings, flush=True)
        # Apply code idioms on tokenizer
        additional_special_tokens = []
        for i in range(args.top_k):
            additional_special_tokens.append(get_idiom_token(i, args.lang))
#         additional_special_tokens.append('/___END___/')
#         additional_special_tokens.append('[SPLIT]')
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        print('Applied '+str(args.top_k)+' code idioms')
        
        # Apply code idioms on model embedding
        model.resize_token_embeddings(len(tokenizer))

    model=Model(model,config,tokenizer,args)
    
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
#     if args.do_test:
#         checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
#         output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
#         model.load_state_dict(torch.load(output_dir),strict=False)      
#         model.to(args.device)
#         result=evaluate(args, model, tokenizer,args.test_data_file, pool)
#         logger.info("***** Eval results *****")
#         for key in sorted(result.keys()):
#             logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        test(args, model, tokenizer, pool)


    return results
# -

if __name__ == "__main__":
    main()



