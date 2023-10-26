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

import argparse
import logging
import os
import pickle
import random
from itertools import cycle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
from pretrain_model import Model
from tqdm import tqdm

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
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
                 input_tokens,
                 input_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg

def convert_examples_to_features(item):
    #parsing
    js,tokenizer,args,parser,lang=item
    code_tokens,dfg=extract_dataflow(js['function'],parser,lang)
    if args.not_use_dfg:
        dfg = []
#     code_tokens,dfg=extract_dataflow(js['original_string'],parser,lang)
    nl_tokens=tokenizer.tokenize(' '.join(js['docstring_tokens']))
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]

    #truncating
    _truncate_seq_pair(nl_tokens,code_tokens, args.block_size-3)
    source_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg=dfg[:args.block_size-len(source_tokens)]
    source_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
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
    length=len([tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]

    return InputFeatures(source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, lang,file_path='train', block_size=512,pool=None):
        self.args=args
        self.args=tokenizer
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()
           
        langs=sorted(args.lang.split(','))
        file_path,postfix=file_path.split(',')
        cached_features_file = os.path.join('{}'.format(args.output_dir),'lang_'+lang+"_word_size_"+str(world_size)+"_rank_"+str(local_rank)+'_size_'+ str(block_size)+'_'+postfix)
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            if 'train' in postfix and local_rank==0:
                for idx, example in enumerate(self.examples[:10]):
                        logger.info("*** Example ***")
                        logger.info("idx: %s",idx)
                        logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                        logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))   
                        logger.info("position_idx: {}".format(' '.join(map(str, example.position_idx)))) 
                        logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                        logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                       
        else:
            self.examples = []
            LANGUAGE = Language('parser/my-languages.so', lang)
            parser = Parser()
            parser.set_language(LANGUAGE) 
            parser = [parser,dfg_function[lang]]
            filename=os.path.join(os.path.join(file_path,lang),postfix)
            logger.info("Creating features from dataset file at %s", filename)
            
            data=[]
            for idx,js in enumerate(pickle.load(open(filename,'rb'))):
                if len(js['docstring_tokens'])!=0:
                    data.append((js,tokenizer, args,parser,lang))
            if 'train' in postfix: 
                data=[x for idx,x in enumerate(data) if idx%world_size==local_rank]
            percent=0  

            for idx in tqdm(range(len(data))):
                x = data[idx]
                example=convert_examples_to_features(x)  
                self.examples.append(example)
                
            if 'train' in postfix and local_rank==0:
                for idx, example in enumerate(self.examples[:10]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("language: {}".format(lang))
                    logger.info("input_tokens: {}".format([x for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids)))) 
                    logger.info("position_idx: {}".format(' '.join(map(str, example.position_idx)))) 
                    logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                    logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
            logger.warning("  Num examples = %d: %d", local_rank,len(self.examples))
            logger.warning("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.attn_mask=np.zeros((len(self.examples),args.block_size,args.block_size),dtype=np.bool)
        self.tag=[False for x in range(len(self.examples))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):                
        if self.tag[item] is False:
            self.tag[item]=True
            node_index=sum([i>1 for i in self.examples[item].position_idx])
            max_length=sum([i!=1 for i in self.examples[item].position_idx])
            self.attn_mask[item,:node_index,:node_index]=True
            for idx,i in enumerate(self.examples[item].input_ids):
                if i in [0,2]:
                    self.attn_mask[item,idx,:max_length]=True
            for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
                self.attn_mask[item,idx+node_index,a:b]=True
                self.attn_mask[item,a:b,idx+node_index]=True
            for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(self.examples[item].position_idx):
                        self.attn_mask[item,idx+node_index,a+node_index]=True            
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.attn_mask[item]))


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,pool=None):
    datasets=[]
    for lang in args.lang.split(','):
        datasets.append( TextDataset(tokenizer, args, lang, file_path=args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file),block_size=args.block_size,pool=pool) )
    return datasets


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




def train(args, train_datasets, model, tokenizer,pool):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
    t_total = args.max_steps
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))    
    if args.local_rank == 0:
        torch.distributed.barrier()         
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", sum([len(train_dataset) for train_dataset in train_datasets])* (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_mlm_loss,tr_man_loss,logging_mlm_loss,logging_man_loss = 0.0, 0.0,0.0,0,0,0,0,0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    probs=[len(x) for x in train_datasets]
    probs=[x/sum(probs) for x in probs]
    probs=[x**0.7 for x in probs]
    probs=[x/sum(probs) for x in probs]
    while True: 
        train_dataloader=np.random.choice(train_dataloaders, 1, p=probs)[0]
        step+=1
        batch=next(train_dataloader)
        inputs_ids, position_idx, attn_mask= (x.to(args.device) for x in batch) 
        model.train()
        mlm_loss,man_loss = model(inputs_ids, position_idx, attn_mask)

        if args.n_gpu > 1:
            loss = mlm_loss.mean()+man_loss.mean()  # mean() to average on multi-gpu parallel training
            mlm_loss = mlm_loss.mean()
            man_loss = man_loss.mean()
        else:
            loss = mlm_loss+man_loss
            
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            mlm_loss = mlm_loss / args.gradient_accumulation_steps
            man_loss = man_loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        tr_mlm_loss+=mlm_loss.item()
        tr_man_loss+=man_loss.item()



        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            output_flag=True
            avg_loss=round((tr_loss - logging_loss) /(global_step- tr_nb),6)
            avg_mlm_loss=round((tr_mlm_loss - logging_mlm_loss) /(global_step- tr_nb),6)
            avg_man_loss=round((tr_man_loss - logging_man_loss) /(global_step- tr_nb),6)
            if global_step %100 == 0:
                logger.info("  steps: %s loss: %s mlm_loss: %s man_loss: %s", global_step, round(avg_loss,6), round(avg_mlm_loss,6), round(avg_man_loss,6))
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                logging_mlm_loss = tr_mlm_loss
                logging_man_loss = tr_man_loss
                tr_nb=global_step

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,6))                    
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['loss'],6)))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module.encoder if hasattr(model,'module') else model.encoder  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                # _rotate_checkpoints(args, checkpoint_prefix)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save.save_pretrained(last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                break


def evaluate(args, model, tokenizer, prefix="",pool=None,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_datasets = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_samplers = [SequentialSampler(eval_dataset) for eval_dataset in eval_datasets]
    eval_dataloaders = [DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size) for eval_dataset,eval_sampler in zip(eval_datasets,eval_samplers)]

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_mlm_loss = 0.0
    eval_man_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for eval_dataloader in eval_dataloaders:
        for batch in eval_dataloader:
            inputs_ids, position_idx, attn_mask= (x.to(args.device) for x in batch) 
            with torch.no_grad():
                mlm_loss,man_loss = model(inputs_ids, position_idx, attn_mask)
                if args.n_gpu > 1:
                    mlm_loss = mlm_loss.mean()
                    man_loss = man_loss.mean()
                eval_loss += mlm_loss.item()+ man_loss.item()
                eval_mlm_loss += mlm_loss.item()
                eval_man_loss += man_loss.item()
            nb_eval_steps += 1

    result = {
        "loss": eval_loss / nb_eval_steps,
        "mlm_loss": eval_mlm_loss / nb_eval_steps,
         "man_loss": eval_man_loss / nb_eval_steps,
    }

    return result


def main():
    os.environ["NCCL_DEBUG"] = "INFO"
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--man_probability", type=float, default=0.2,
                        help="Ratio of tokens to mask for masked language modeling loss")
    
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")


    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="For distributed training: local_rank")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="For distributed training: local_rank")     
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--lang', type=str)
    parser.add_argument('--not_use_dfg', action='store_true',
                        help="disable data flow")    
    pool = None
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank+=args.node_index*args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    #args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)   
    else:
        model = RobertaForMaskedLM(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # Training
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    global_step, tr_loss = train(args, train_dataset, model, tokenizer,pool)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == "__main__":
    main()




