# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import json
import argparse
import logging
import os
import pickle
import random
from itertools import cycle
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from codesearch_model import Model
from tqdm import tqdm

from code_idioms.basic_utils import get_idiom_token, get_idiom_path, dump_pkl, load_pkl

# +
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
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

def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


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
    
    return InputFeatures(source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'],idiom_to_code)



# +
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None, idiom_type=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
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
        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids))


# -

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)




def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool, idiom_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool, idiom_type='codebase')
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        attn_mask = batch[1].to(args.device)
        position_idx =batch[2].to(args.device)
        with torch.no_grad():
            code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    rank_logs = {}
    top1_preds = {}
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        top1_preds[url] = code_urls[sort_id[0]]
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
            rank_logs[url] = rank
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }
    with open(args.output_dir+'/test_rank.pkl', 'wb') as fp:
        pickle.dump(rank_logs, fp, pickle.HIGHEST_PROTOCOL)
    with open(args.output_dir+'/top1_preds.pkl', 'wb') as fp:
        pickle.dump(top1_preds, fp, pickle.HIGHEST_PROTOCOL)

    return result


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
    set_seed(args)

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
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results

if __name__ == "__main__":
    main()
