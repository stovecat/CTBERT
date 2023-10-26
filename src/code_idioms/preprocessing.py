import argparse
import logging
import os
import sys
import random
import torch
import json
import numpy as np
#from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

from tqdm import tqdm, trange
import multiprocessing

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser

try:
    from basic_utils import load_jsonl, save_jsonl, load_pkl, dump_pkl, get_idiom_path, is_none, is_false
    from ast2code import *
except ModuleNotFoundError:
    from code_idioms.basic_utils import load_jsonl, load_pkl, dump_pkl, get_idiom_token, is_none, is_false
    from code_idioms.ast2code import *

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

tree_dict = None

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('./parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

def requires_preprocessing(path):    
    if os.path.exists(path):
        return False
    else:
        return True

##########################################################################################
# LOAD functions
# #########################################################################################    

def load_tree_dict(args, pool, data=None):
    tree_path = args.idiom_path+'/tree_dict.pkl'
    if not os.path.exists(tree_path):
        assert data is not None
        # STEP 1
        print('STEP 1: Convert codes into ASTs', flush=True)
        if args.multiprocess:
            tree_dict = pool.map(convert_examples_to_dict, tqdm(data,total=len(data)))
        else:
            data_size = len(data)
            if args.lang == 'javascript':
                with open(tree_path, 'w') as fp:
                    for _ in tqdm(range(data_size)):
                        json.dump(convert_examples_to_dict(data.pop(0)), fp)
                        fp.write('\n')
            else:
                tree_dict = []
                for _ in tqdm(range(data_size)):
                    tree_dict.append(convert_examples_to_dict(data.pop(0)))
        if args.save_type == 'pickle':
            dump_pkl(tree_path, tree_dict)
        else:
            if args.lang == 'javascript':
                pass
            else:
                save_jsonl(tree_path, tree_dict)
        print('tree_dict is saved. please re-run the preprocessing.',flush=True)
        exit()
    else:
        if args.save_type == 'pickle':
            tree_dict = load_pkl(tree_path)
        else:
            tree_dict = load_jsonl(tree_path)
    return tree_dict

def load_pTSG(args, tree_dict):
    pTSG_path = args.data_path+'/pTSG.pkl'
    if not os.path.exists(pTSG_path):
        # STEP 2
        print('STEP 2: Obtain pTSG', flush=True)
        pTSG = get_pTSG(tree_dict)
        dump_pkl(pTSG_path, pTSG)
    else:
        pTSG = load_pkl(pTSG_path)
    return pTSG

def load_tree_frags(args, pool, tree_dict):
    T = args.ptsg_threshold
    tree_frags_path = args.data_path+'/tree_frags.pkl'
    if not os.path.exists(tree_frags_path):
        pTSG = load_pTSG(args, tree_dict)
        # STEP 3
        print('STEP 3: Build tree fragments by threshold '+str(T), flush=True)
        tree_frags = split_frags(pool, tree_dict, pTSG, threshold=T)
        if args.save_type == 'pickle':
            dump_pkl(tree_frags_path, tree_frags)
        else:
            save_jsonl(tree_frags_path, tree_frags)
    else:
        if args.save_type == 'pickle':
            tree_frags = load_pkl(tree_frags_path)            
        else:
            tree_frags = load_jsonl(tree_frags_path)
    return tree_frags

def load_ranked_frags(args, pool, data):
    ranked_frags_path = args.data_path+'/ranked_frags.pkl'
    if not os.path.exists(ranked_frags_path):
        global tree_dict
        tree_frags = load_tree_frags(args, pool, tree_dict)
        # STEP 4
        print('STEP 4: Rank fragments', flush=True)
        ranked_frags = rank_frags(tree_frags)
        if args.save_type == 'pickle':
            dump_pkl(ranked_frags_path, ranked_frags)
        else:
            save_jsonl(ranked_frags_path, ranked_frags)
    else:
        if args.save_type == 'pickle':
            ranked_frags = load_pkl(ranked_frags_path)
        else:
            ranked_frags = load_jsonl(ranked_frags_path)
    return ranked_frags

def load_filtered_frags(args, pool, data):
    filtered_frags_path = args.data_path+'/filtered_frags.pkl'
    if not os.path.exists(filtered_frags_path):
        ranked_frags = load_ranked_frags(args, pool, data)
        # STEP 5
        print('STEP 5: Filter fragments by tree depth', flush=True)
        filtered_frags = filter_frags(args, pool, ranked_frags)
        if args.save_type == 'pickle':
            dump_pkl(filtered_frags_path, filtered_frags)
        else:
            save_jsonl(filtered_frags_path, filtered_frags)
    else:
        if args.save_type == 'pickle':
            filtered_frags = load_pkl(filtered_frags_path)
        else:
            filtered_frags = load_jsonl(filtered_frags_path)
    return filtered_frags

def load_top_k_frags(args, pool, data=None):
    K = args.top_k
    top_k_frags_path = args.data_path+'/top_k_frags.pkl'
    if not os.path.exists(top_k_frags_path):
        assert data is not None
        global tree_dict
        if tree_dict is None:
            tree_dict = load_tree_dict(args, pool, data)
        filtered_frags = load_filtered_frags(args, pool, data)
        # STEP 6
        print('STEP 6: Truncate top '+str(K)+' frags', flush=True)
        top_k_frags = [v[0] for v in filtered_frags[:K]]
        if args.save_type == 'pickle':
            dump_pkl(top_k_frags_path, top_k_frags)
        else:
            save_jsonl(top_k_frags_path, top_k_frags)
    else:
        if args.save_type == 'pickle':
            top_k_frags = load_pkl(top_k_frags_path)
        else:
            top_k_frags = load_jsonl(top_k_frags_path)
    return top_k_frags

def load_code_idioms(args, pool, data, tokenizer, train_data=None):
    if args.idiom_loss == 'span':
        code_idioms_path = args.idiom_path+'/code_idioms.pkl'
    elif args.idiom_loss == 'full':
        code_idioms_path = args.idiom_path+'/code_idioms_full.pkl'
    else:
        raise NotImplementedError
    if not os.path.exists(code_idioms_path):
        global tree_dict
        if tree_dict is None:
            tree_dict = load_tree_dict(args, pool, data)
        top_k_frags = load_top_k_frags(args, pool, train_data)
        # STEP 6
        print('STEP 7: Extract code idioms', flush=True)
        code_idioms = extract_code_idioms(pool, top_k_frags, [d[0] for d in data], tokenizer, args)
        dump_pkl(code_idioms_path, code_idioms)
    else:
        code_idioms = load_pkl(code_idioms_path)
    return code_idioms
##########################################################################################
# STEP 1: Convert codes into ASTs 
# #########################################################################################
def tree2dict(tree, code):
    max_recursion = 10 ** 9
    #sys.setrecursionlimit(max_recursion)
    current_recursion = 0
    def inner_loop(_cursor, mr, cr, code):
        cr += 1
        if cr >= mr:
            #print('Max recursion error!', flush=True)
#             dump_pkl('treedict_max_recursion.pkl', code)
            return False
        
        if not _cursor.node.is_named:
            return None
        cursor = _cursor.node.walk()
        child_cnt = cursor.node.named_child_count
        child_cnt = cursor.node.child_count
        if not cursor.node.is_named and _cursor.node.named_child_count > 0:
            print("Language dependency?")
#             print(cursor.node.type)
            assert 1 == 2
        tmp_dict = {'type': cursor.node.type, 
                    'start_point': cursor.node.start_point, 
                    'end_point': cursor.node.end_point
                   }                
        if child_cnt == 0:
            return tmp_dict
        
        tmp_dict['children'] = []
        cursor.goto_first_child()
        child_dict = inner_loop(cursor, mr, cr, code)
        if is_false(child_dict):
            return False
        if child_dict is not None:
            tmp_dict['children'].append(child_dict)
        
        if child_cnt > 1:
            for i in range(child_cnt-1):
                if cursor.node.next_sibling is None:
                    break
                cursor = cursor.node.next_sibling.walk()
                child_dict = inner_loop(cursor, mr, cr, code)
                if is_false(child_dict):
                    return False
                if child_dict is not None:
                    tmp_dict['children'].append(child_dict)
        return tmp_dict
    cursor = tree.root_node.walk()
    return inner_loop(cursor, max_recursion, current_recursion, code)

def convert_examples_to_dict(item):
    def extract_treedict(code, parser,lang):
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
            td = tree2dict(tree, code)
            if is_false(td):
                return None
            return td
        except:
            print('error while parsing',flush=True)
            assert 1 == 2
        return None
    js,tokenizer,args=item
#     print("args.lang", args.lang)
    parser=parsers[args.lang]
    tree_dict=extract_treedict(js[args.raw_code_key],parser,args.lang)
    return tree_dict

##########################################################################################
# STEP 2: Obtain pTSG 
# #########################################################################################
def _get_pTSG(node, parent=None, pTSG=None):
    if pTSG is None:
        pTSG = {}
    # Skip list
    if type(node) is list:
        for _node in node:
            pTSG = _get_pTSG(_node, parent, pTSG)
        return pTSG
    if type(node) is not dict:
        return pTSG
    # Node is dict type
    assert type(node) is dict
    child = node['type']
    if parent is None:
        pass
    else:
        # define parent in pTSG
        if parent not in pTSG.keys():
            pTSG[parent] = {}
        if child not in pTSG[parent].keys():
            pTSG[parent][child] = 0
        pTSG[parent][child] += 1
    if 'children' in node.keys():
        pTSG = _get_pTSG(node['children'], child, pTSG)
    return pTSG

def get_pTSG(D):
    pTSG = None
    for d in D:
        if is_none(d):
            continue
        pTSG = _get_pTSG(d, pTSG=pTSG)
    for p in pTSG.keys():
        cnt = 0
        for c in pTSG[p].keys():
            cnt += pTSG[p][c]
        for c in pTSG[p].keys():
            pTSG[p][c] /= cnt
    return pTSG

##########################################################################################
# STEP 3: Build && count unique tree fragments by threshold T
# #########################################################################################
import copy
import math

def update(d, key_lst , val):
    for k in key_lst[:-1]:
        d = d[k]
    d[key_lst[-1]] = val

# Get parent node
def get_T_s(T, trace):
    update(T, trace, 'HOLE')
    return T


def _split_frags(data):
    z, pTSG, threshold = data
    assert type(z) is dict
    T = copy.deepcopy(z)
    frags = []
    
    def inner_loop(T, trace, _frags, parent):
        if type(T) is dict:
            del T['start_point']
            del T['end_point']
            current_name = T['type']
            key = 'children'
            if key in T.keys():
                T[key], frags = inner_loop(T[key], trace+[key], _frags, current_name)
            if pTSG[parent][current_name] < threshold:
                _frags.append(T)
                return 'HOLE', _frags

        elif type(T) is list:
            for i in range(len(T)):
                T[i], _frags = inner_loop(T[i], trace+[i], _frags, parent)
            
        return T, _frags

    key = 'children'
    if key in T.keys():
        del T['start_point']
        del T['end_point']
        trace = [key]
        parent = T['type']
        T[key], frags = inner_loop(T[key], trace, frags, parent)
    del T            
    return frags


def is_same_frag(T1, T2):
    # only one node has HOLE
    if ((type(T1) is str and T1 == 'HOLE') and not (type(T2) is str and T2 == 'HOLE')) or \
        ((type(T2) is str and T2 == 'HOLE') and not (type(T1) is str and T1 == 'HOLE')):
        return False
    
    # both nodes have HOLE
    if (type(T1) is str and T1 == 'HOLE') and (type(T2) is str and T2 == 'HOLE'):
        return True
    
    # both the node types should be the same
    if (T1['type'] != T2['type']):
        return False

    assert type(T1) is dict and type(T2) is dict
    
    child = 'children'
    if child in T1.keys():
        if child in T2.keys():
            if type(T1[child]) is dict and type(T2[child]) is dict:
                if not is_same_frag(T1[child], T2[child]):
                    # different child
                    return False
            elif type(T1[child]) is list and type(T2[child]) is list:
                # both 
                if len(T1[child]) != len(T2[child]):
                    # different num of children
                    return False
                for i in range(len(T1[child])):
                    if not is_same_frag(T1[child][i], T2[child][i]):
                        # different child
                        return False
            else:
                # One has child (a node), but another has children (node list)
                return False
        else:
            # only T1 has children
            return False
    elif child in T2.keys():
        # only T2 has children
        return False
        
    return True


# Drop duplicates
def drop_dup_frags(tree_frags, tree_frags_2=None, tqdm_bar=False):
    iterate = None
    if tqdm_bar:
        iterate = tqdm(range(len(tree_frags)))
    else:
        iterate = range(len(tree_frags))
            
    result = []
    # tree_frags_2 should be a set of unique frags
    if tree_frags_2 is not None:
        result = tree_frags_2
    
    for i in iterate:
        tf = tree_frags[i]
        if len(result) == 0:
            result.append(tf)
            continue

        flag = True
        for rtf in result:
            if is_same_frag(tf, rtf):
                flag = False
                break
        if flag:
            result.append(tf)
    return result

def pool_is_same_frag(data):
    T1, T2 = data
    return is_same_frag(T1, T2)

def drop_dup_frags_with_cnt(tree_frags, tree_frags_2=None, tqdm_bar=False, pool=None):
    global cpu_cont
    result = []
    
    # tree_frags_2 should be a set of unique frags
    if tree_frags_2 is not None:
        result = tree_frags_2
    
    iterator = range(len(tree_frags))
    if tqdm_bar:
        iterator = tqdm(iterator)

    for i in iterator:
        if type(tree_frags[i]) is list:
            tf = tree_frags[i][0]
            cnt = tree_frags[i][1]
        else:
            tf = tree_frags[i]
            cnt = 1
        if len(result) == 0:
            result.append([tf, cnt])
            continue
        
        flag = True
        if pool is None:
            for j, (rtf, cnt) in enumerate(result):
                if is_same_frag(tf, rtf):
                    flag = False
                    result[j][1] += 1
                    break
        else:
            for j in range(math.ceil(len(result)/cpu_cont)):
                data = []
                for k in range(cpu_cont):
                    tmp_idx = j*cpu_cont + k
                    if tmp_idx >= len(result):
                        break
                    data.append((tf, result[tmp_idx][0]))
                if args.multiprocess:
                    r = pool.map_async(pool_is_same_frag, data)
                    r.wait()
                    tmp_result = r.get()
                else:
                    tmp_result = []
                    data_size = len(data)
                    for _ in range(data_size):
                        tmp_result.append(pool_is_same_frag(data.pop(0)))
                del data
                if True in tmp_result:
                    flag = False
                    result[j][1] += 1
                    break
            
        if flag:
            result.append([tf, cnt])
    return result


def multi_drop_dup_frags(data):
    tree_frags, tree_frags_2, tqdm_bar = data
    #return drop_dup_frags(tree_frags, tree_frags_2, tqdm_bar)
    return drop_dup_frags_with_cnt(tree_frags, tree_frags_2, tqdm_bar)

def inner_drop_dup_frags(pool, data):
    global cpu_cont
    result = []
    iterator = data
    tqdm_bar = True
    if len(data) > cpu_cont:
        iterator = tqdm(iterator)
        tqdm_bar = False
    for tree_frags, tree_frags_2, _ in iterator:
        result.append(drop_dup_frags_with_cnt(tree_frags, tree_frags_2, tqdm_bar, pool))
    return result

def split_frags(pool, D, pTSG, threshold=0.2):
    data = []
    for d in D:
        if is_none(d):
            continue
        data.append((d, pTSG, threshold))
    del D
    if args.multiprocess:
        nested_tree_frags = pool.map(_split_frags, tqdm(data,total=len(data)))
    else:
        nested_tree_frags = []
        data_size = len(data)
        for _ in tqdm(range(data_size)):
            nested_tree_frags.append(_split_frags(data.pop(0)))
    del data
    ntf_size = len(nested_tree_frags)
    data = []
    for _ in range(ntf_size):
        data.append((nested_tree_frags.pop(0), None, False))
    if args.multiprocess:
        r = pool.map_async(multi_drop_dup_frags, tqdm(data,total=len(data)), chunksize=1)
        r.wait()
        nested_tree_frags = r.get()
    else:
        nested_tree_frags = []
        data_size = len(data)
        for _ in tqdm(range(data_size)):
            nested_tree_frags.append(multi_drop_dup_frags(data.pop(0)))
    del data
    for idx in tqdm(range(math.ceil(math.log(len(nested_tree_frags), 2)))):
        data = []
        for idx2 in range(math.ceil(len(nested_tree_frags)/2)):
            if idx2*2 == len(nested_tree_frags)-1:
                data.append((nested_tree_frags[idx2*2], None, False))
            else:
                data.append((nested_tree_frags[idx2*2], nested_tree_frags[idx2*2+1], False))
        if args.multiprocess:
            r = pool.map_async(multi_drop_dup_frags, tqdm(data,total=len(data)), chunksize=1)
            r.wait()
            nested_tree_frags = r.get()
        else:
            nested_tree_frags = []
            data_size = len(data)
            for _ in tqdm(range(data_size)):
                nested_tree_frags.append(multi_drop_dup_frags(data.pop(0)))
        del data
        
        if len(nested_tree_frags) == 1:
            nested_tree_frags = nested_tree_frags[0]
            break
    return nested_tree_frags

##########################################################################################
# STEP 4: Rank fragments
# #########################################################################################
def rank_frags(tree_frags):
    return sorted(tree_frags, key=lambda x: x[1], reverse=True)

##########################################################################################
# STEP 5: Filter fragments by tree depth
# #########################################################################################
def get_depth(T):
    def inner_loop(T):
        cnt_list = []
        if type(T) is dict:
            child = 'children'
            if child in T.keys():
                cnt_list.append(inner_loop(T[child]))
            if cnt_list == []:
                return 1            
            return max(cnt_list) + 1
        elif type(T) is list:
            for i in range(len(T)):
                cnt_list.append(inner_loop(T[i]))
            if cnt_list == []:
                return 1
            return max(cnt_list)
        elif type(T) is str and T == 'HOLE':
            return 1
        else:
            return 0

    return inner_loop(T)

def get_num_nodes(T):
    def inner_loop(T):
        cnt_list = []
        if type(T) is dict:
            child = 'children'
            if child in T.keys():
                cnt_list.append(inner_loop(T[child]))
            if cnt_list == []:
                return 1            
            return sum(cnt_list) + 1
        elif type(T) is list:
            for i in range(len(T)):
                cnt_list.append(inner_loop(T[i]))
            if cnt_list == []:
                return 1
            return sum(cnt_list)
        elif type(T) is str and T == 'HOLE':
            return 1
        else:
            return 0
    return inner_loop(T)

def filter_frags(args, pool, tree_frags):
    F = args.frequency_threshold * args.corpus_size
    D = args.depth_threshold
    N = args.num_of_nodes_threshold
    #TBD run by multiprocessing
    filtered_frags = []
    for i in tqdm(range(len(tree_frags))):
        tf, cnt = tree_frags[i]
        if cnt < F:
            break
        if get_depth(tf) > D and get_num_nodes(tf) > N:
            filtered_frags.append((tf, cnt))
    return filtered_frags

def contained(T1, T2):
    # T1 has HOLE
    if (type(T1) is str and T1 == 'HOLE'):
        return True
    
    # only T2 has HOLE
    if (type(T2) is str and T2 == 'HOLE'):
        return False

    if (T1['type'] != T2['type']):
        return False

    child = 'children'
    if child in T1.keys():
        if child in T2.keys():
            if type(T1[child]) is dict and type(T2[child]) is dict:
                # both childs are nodes
                if not contained(T1[child], T2[child]):
                    # not the same
                    return False
            elif type(T1[child]) is list and type(T2[child]) is list:
                if len(T1[child]) == 0:
                    return True
                # both children are lists
                if len(T1[child]) > len(T2[child]):
                    # T1[child]'s num of children is bigger than T2[child]'s
                    return False
                for i in range(len(T1[child])):
                    if not contained(T1[child][i], T2[child][i]):
                        return False
            else:
                # both T1 T2 have children but different types
                return False
        else:
            # only T1 has children
            return False
    elif child in T2.keys():
        # only T2 has children
        return True
    return True

def find_idiom_tree(T, D):
    result = []
    if type(D) is dict:
        if contained(T, D):
            result.append(D)
        child = 'children'
        if child in D.keys():
            _result = find_idiom_tree(T, D[child])
            if len(_result) > 0:
                result.extend(_result)
    elif type(D) is list:
        result = []
        for d in D:
            _result = find_idiom_tree(T, d)
            if len(_result) > 0:
                result.extend(_result)
    return result

def apply_hole(idiom, tf):
    def inner_loop(T1, T2):
        if (type(T1) is str and T1 == 'HOLE'):
            return 'HOLE'
        
        child = 'children'
        if type(T2) is dict and (child not in T2.keys() or len(T2[child]) == 0):
            return T2
        if type(T2) is dict:
            result = copy.deepcopy(T2)
            result['children'] = inner_loop(T1[child], T2[child])
            return result            
        elif type(T2) is list:
            children = []
            for i in range(len(T1)):
                children.append(inner_loop(T1[i], T2[i]))
            return children
    return inner_loop(idiom, tf)


# def get_tokens_with_idioms(code, tree, tokens, types, lang):
#     ast_tree = PARSER.parse(bytes(code, "utf8"))
#     node = ast_tree.root_node
#     def find_root(node, tree):
#         #TBD find real frag root
#         if node.start_point == tree['start_point'] and node.type == tree['type']:
#             return node
#         else:
#             for child in node.children:
#                 result = find_root(child, tree)
#                 if result is not None:
#                     return result
#             return None
#     def _get_tokens(node, tree, tokens, types):
#         if node.children is None:
#             assert 1 == 2
#         if not node.is_named:
#             tokens.append([node.start_point, node.end_point])
#             types.append(str(node.type))
#         elif len(node.children) == 0:
#             tokens.append([node.start_point, node.end_point])
#             types.append(str(node.type))
#         if (
#             str(node.type) not in ["concatenated_string", "string_array", "chained_string"]
#             and "string" in str(node.type)
#             or "char" in str(node.type)
#         ):
#             tokens.append([node.children[0].start_point, node.children[-1].end_point])
#             types.append(str(node.type))
#             return
#         if tree is not None and 'children' in tree.keys():
#             cnt = 0
#             for child in node.children:
#                 if cnt >= len(tree['children']):
#                     if not child.is_named:
#                         _get_tokens(child, None, tokens, types)
#                     else:
#                         continue
#                 else:
#                     if tree['children'][cnt] == 'HOLE':
#                         cnt += 1
#                         continue
                        
#                     if not child.is_named:
#                         _get_tokens(child, None, tokens, types)
#                     else:
#                         if child.type == tree['children'][cnt]['type']:
#                             _get_tokens(child, tree['children'][cnt], tokens, types)
#                             cnt += 1
#     root = find_root(node, tree)
#     _get_tokens(root, tree, tokens, types)
    
def get_raw_codes(tokens, types, code, _print=False):
    result = []
    for r, t in zip(tokens, types):
        c = code.split('\n')
        c[r[1][0]] = c[r[1][0]][:r[1][1]]
        c[r[0][0]] = c[r[0][0]][r[0][1]:]
        c = c[r[0][0]:r[1][0]+1]
        if _print:
            print(r)
            print(t)
            print("\n".join(c))
            print()
        result.append((r, t, "\n".join(c)))
    return result

def get_idiom_code(full, idiom):
    start_point = idiom[0][0]
    end_point = idiom[-1][0]
    
    start_idx = None
    end_idx = None
    for i, f in enumerate(full):
        if f[0] == start_point:
            start_idx = i
        elif f[0] == end_point:
            end_idx = i
    
    result = copy.deepcopy(full)[start_idx:end_idx]
    cnt = 0
    for i in range(len(result)):
        if idiom[cnt][0] == result[i][0]:
            cnt += 1
        else:
            result[i] = 'HOLE'
    return result, full[start_idx:end_idx]
########################################################################################## 
# STEP 6: Truncate top K frags
# ######################################################################################### 
# No function is needed

########################################################################################## 
# STEP 7: Extract code idioms
# ######################################################################################### 
def extract_code_idioms(pool, top_k_frags, data, tokenizer, args):
    global tree_dict
    
    items = []
#     for idx in range(len(tree_dict)):
#         items.append((data[idx], tree_dict[idx], top_k_frags))
#     r = pool.map_async(_extract_code_idioms, tqdm(items,total=len(items)))
    tree_dict_size = len(tree_dict)
    for idx in range(tree_dict_size):
        items.append((data.pop(0), tree_dict.pop(0), top_k_frags, tokenizer, False, args))
    if args.multiprocess:
        r = pool.map_async(_extract_code_idioms_v2, tqdm(items,total=len(items)))
        r.wait()
        idiom_span_list = r.get()
    else:
        idiom_span_list = []
        items_size = len(items)
        for _ in tqdm(range(items_size)):
            idiom_span_list.append(_extract_code_idioms_v2(items.pop(0)))
    
    return idiom_span_list

# def _extract_code_idioms(items):
#     d, td, top_k_frags = items
#     idiom_list = []
#     for idiom_idx in range(len(top_k_frags)):
#         tree_list = find_idiom_tree(top_k_frags[idiom_idx], td)
#         if len(tree_list) > 0:
#             idiom_list.append((idiom_idx, tree_list))
#     try:
#         code = remove_comments_and_docstrings(d['function'], 'python')
#     except:
#         code = d[idx]['function']

#     root = PARSER.parse(bytes(code, "utf8")).root_node
#     full_tokens = []
#     full_types = []
#     get_tokens(root, full_tokens, full_types)
#     full_result = get_raw_codes(full_tokens, full_types, code)
#     span_list = []
#     for idiom_idx, tree_list in idiom_list:
#         for tree in tree_list:
#             _idiom = apply_hole(top_k_frags[idiom_idx], tree)
#             # Full code => node
#             tokens = []
#             types = []
#             get_tokens_with_idioms(code, _idiom, tokens, types)
#             result = get_raw_codes(tokens, types, code)
#             if len(result) == 0:
#                 continue
#             span_list.append((idiom_idx, (result[0][0][0], result[-1][0][-1])))

#     sorted_span_list = [ v[:-1] for v in \
#                         sorted([(r[0], r[1], r[1][0][0]*1e+5 + r[1][0][1]) \
#                                 for r in span_list], key=lambda x: x[-1]) \
#                         ]
#     return tuple(sorted_span_list)
    
def _extract_code_idioms_v2(items):
    d, td, code_idioms, tokenizer, debug, args = items
    if is_none(td):
        return tuple([])
    # Start idx, End idx
#     def get_idiom_code(full, idiom):
#         start_point = idiom[0][0]
#         end_point = idiom[-1][0]

#         start_idx = None
#         end_idx = None
#         for i, f in enumerate(full):
#             if f[0] == start_point:
#                 start_idx = i
#             elif f[0] == end_point:
#                 end_idx = i

#         result = copy.deepcopy(full)[start_idx:end_idx]
#         cnt = 0
#         for i in range(len(result)):
#             if len(idiom) < cnt and idiom[cnt][0] == result[i][0]:
#                 cnt += 1
#             else:
#                 result[i] = 'HOLE'
#         return result, full[start_idx:end_idx]
    
    idiom_list = []
    for idiom_idx in range(len(code_idioms)):
        _td = copy.deepcopy(td)
        _idiom = copy.deepcopy(code_idioms[idiom_idx])
        tree_list = find_idiom_tree(_idiom, _td)
        if len(tree_list) > 0:
            idiom_list.append((idiom_idx, tree_list))
        del _td, _idiom
    code=d[args.raw_code_key]
    if args.lang=="php":
        code="<?php"+code+"?>"
    try:
        code = remove_comments_and_docstrings(code, args.lang)
    except:
        pass
    root = parsers[args.lang][0].parse(bytes(code, "utf8")).root_node
    full_tokens = []
    full_types = []
    get_tokens(root, full_tokens, full_types)
    full_result = get_raw_codes(full_tokens, full_types, code)
    if debug:
        print(code)
        print('full_result:')
        for v in full_result:
            print(v)
    code_tokens=[tokenizer.tokenize('@ '+x[-1])[1:] if idx!=0 else tokenizer.tokenize(x[-1]) for idx,x in enumerate(full_result)]
    if debug:
        print(code_tokens,'\n')
        
    result_set = set()
    result_list = []
    for idiom_idx, tree_list in idiom_list:
        for tree in tree_list:
            _idiom = apply_hole(code_idioms[idiom_idx], tree)
            if args.idiom_loss == 'span':
                # Full code => node
                start_idx = None
                end_idx = None
                for i, v in enumerate(full_result):
                    if tuple(v[0][0]) == tuple(_idiom['start_point']):
                        start_idx = i
                    if tuple(v[0][1]) == tuple(_idiom['end_point']):
                        end_idx = i
                if start_idx is None or end_idx is None:
                    continue
                result = (start_idx, end_idx)
                if result not in result_set:
                    result_list.append((idiom_idx, result))
                    result_set.add(result)
                    if debug:
    #                     global args
                        print(get_idiom_token(idiom_idx, args.lang))
                        print('_idiom:',_idiom)
                        print("start_idx:",start_idx, full_result[start_idx])
                        print("end_idx:",end_idx, full_result[end_idx])
                        print()
            elif args.idiom_loss == 'full':
                def get_leaf_start_end(idiom):
                    _r = []
                    if 'children' in idiom.keys():
                        for child in idiom['children']:
                            if child == 'HOLE':
                                continue
                            elif type(child) == dict:
                                no_hole_flag = True
                                _r.extend(get_leaf_start_end(child))
                            else:
                                raise NotImplementedError
                    if len(_r) == 0:
                        return [(idiom['start_point'], idiom['end_point'])]
                    else:
                        return _r
                result = []
                idiom_points = get_leaf_start_end(_idiom)
                none_flag = False
                for ip in idiom_points:
                    start_idx = None
                    end_idx = None
                    for i, v in enumerate(full_result):
                        if tuple(v[0][0]) == tuple(ip[0]):
                            start_idx = i
                        if tuple(v[0][1]) == tuple(ip[1]):
                            end_idx = i
                    if start_idx is None or end_idx is None:
                        none_flag = True
#                         print(ip,'is not in full code tokens:', full_result)
                        break
                    result.append((start_idx, end_idx))
                if none_flag:
                    continue
                if result not in result_list:
                    result_list.append((idiom_idx, result))
            else:
                raise NotImplementedError
            
    sorted_result_list = sorted(result_list, key=lambda x: x[1][0])
    return tuple(sorted_result_list)
    
def apply_idiom(_full, idiom_list):
    full = []
    in_idiom = None
    for idx in range(len(_full)):
        flag = True
        for idiom_idx, idiom in idiom_list:
            start_point = idiom[0][0]
            end_point = idiom[-1][0]
            
            start_idx = None
            end_idx = None
            for i, f in enumerate(_full):
                if f[0] == start_point:
                    start_idx = i
                elif f[0] == end_point:
                    end_idx = i
            
            if start_idx is None or end_idx is None:
                continue

            if idx >= start_idx and idx <= end_idx:
                # all idioms should be disjoint
                if in_idiom is not None:
                    if in_idiom != idiom_idx:
                        continue
                if not flag:
                    continue
                assert flag
                flag = False
                if idx == start_idx:
                    full.append(list(_full[idx]))
                    full[-1][1] = get_idiom_token(idiom_idx, lang)
                    full[-1][2] = get_idiom_token(idiom_idx, lang)
                    in_idiom = idiom_idx
                else:
                    if in_idiom is None:
                        flag = True
                        break
                    if idx == end_idx:
                        full.append(list(_full[idx]))
                        full[-1][1] = '/___END___/'
                        full[-1][2] = '/___END___/'
                        assert in_idiom == idiom_idx
                        in_idiom = None
                    elif _full[idx][0] not in [v[0] for v in idiom]:
                        full.append(_full[idx])
        if flag:
            full.append(_full[idx])
    return full

def print_idiom(idiom_code, merge_hole=False, show_type=False):
    prev_sl, prev_sp, prev_el, prev_ep = None, None, None, None
    prev_hole = False
    in_idiom = False
    result = [""]
    for v in idiom_code:
            
        if v == 'HOLE':
            if merge_hole and prev_hole:
                continue
            result[-1] += v
            prev_hole = True
            prev_sl, prev_sp, prev_el, prev_ep = None, None, None, None
            continue
        position, t, c = v
        cur_sl, cur_sp = position[0]
        cur_el, cur_ep = position[1]
        if prev_sl is None:
            if not prev_hole:
                for _ in range(cur_sp):
                    result[-1] += ' '
        else:
            if not in_idiom and prev_el < cur_sl:
                result.append("")
                for _ in range(cur_sp):
                    result[-1] += ' '
            space = cur_sp - prev_ep
            if prev_el != cur_sl:
                result[-1] += ' '
            if in_idiom and (space > 1 or (result[-1][-1] == ',')):
                if result[-1][-4:] != '___/' and v[-1] != '/___END___/':
                    result[-1] += '[SPLIT]'
            elif v[-1] != '/___END___/':
                for _ in range(space):
                    result[-1] += ' '
        if '/___IDIOM' in v[-1] or v[-1] == '/___END___/':
            in_idiom = not in_idiom
            if prev_el is None:
                result[-1] += v[-1]
            elif prev_el < cur_sl:
                result[-1] += v[-1]
            else:
                result[-1] += v[-1]
            prev_sl, prev_sp, prev_el, prev_ep = cur_sl, cur_sp, cur_el, cur_ep
            continue
        if show_type:
            result[-1] += t
        else:
            result[-1] += c

        prev_sl, prev_sp, prev_el, prev_ep = cur_sl, cur_sp, cur_el, cur_ep
        prev_hole = False
    return "\n".join(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file")
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
    parser.add_argument('--raw_code_key', type=str, default='function',
                        help="Raw code key of the data dictionaries")
    parser.add_argument('--idiom_loss', type=str, default='span',
                        help="Idiom loss type")
    parser.add_argument('--multiprocess', action='store_true',
                        help="Use multiprocessing")    
    parser.add_argument('--save_type', type=str, default='jsonl',
                        help="Save type for big files")    
    
    def get_tokenizer(model_name='microsoft/graphcodebert-base'):
        tokenizer_name = model_name
        config = RobertaConfig.from_pretrained(model_name)
        return RobertaTokenizer.from_pretrained(tokenizer_name)
    
    def pack_data(data, tokenizer, args):
        args.corpus_size = len(data)
        items=[]
        for _ in range(args.corpus_size):
            items.append((data.pop(0),tokenizer,args))
        return items
    
    cpu_cont = 16
    tree_dict = None
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()
    sys.setrecursionlimit(10 ** 9)
    
    #Generate folders if not exist
    idiom_path = get_idiom_path(args, '', True)
    data_path = get_idiom_path(args, 'ast', True)
    args.idiom_path = './'+args.data_path+'/'+idiom_path+'/'+"".join(args.target_data_file.split('/')[-1].split('.')[:-1])
    args.data_path = './'+args.data_path+'/'+data_path+'/ast'
    
#     args.data_path = './'+args.data_path+\
#                      '/T:'+str(args.ptsg_threshold)+\
#                      '_F:'+str(args.frequency_threshold)+\
#                      '_D:'+str(args.depth_threshold)+\
#                      '_N:'+str(args.num_of_nodes_threshold)+\
#                      '_K:'+str(args.top_k)+'/'
#     args.idiom_path = args.data_path+'.'.join("".join(args.target_data_file.split('../')).split('.')[:-1])
#     args.data_path += "/".join(args.train_data_file.split('/')[1:-2])+'/'+args.lang+'/ast/'
    from pathlib import Path
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    Path(args.idiom_path).mkdir(parents=True, exist_ok=True)
    
    tokenizer = get_tokenizer()
    target_data = load_pkl(args.target_data_file)
    items = pack_data(target_data, tokenizer, args)
#     print('Mining Code Idioms and Apply to the Training dataset.',flush=True)
    print('Code idiom mining for',args.lang,'language', flush=True)
    if not os.path.exists(args.data_path+'/top_k_frags.pkl'):
        train_data = load_pkl(args.train_data_file)
        train_items = pack_data(train_data, tokenizer, args)
        code_idioms = load_code_idioms(args, pool, items, tokenizer, train_items)
    else:
        code_idioms = load_code_idioms(args, pool, items, tokenizer)
    for idx, v in enumerate(code_idioms[:3]):
        print('Index', str(idx)+':', v, '\n')
    pool.close()
    

            
