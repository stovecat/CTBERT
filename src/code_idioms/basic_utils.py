import json
import dill as pickle

def load_jsonl(path, args=None, _range=None):
    data=[]
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if _range is not None:
                if _range[0] <= idx and idx < _range[1]:
                    pass
                else:
                    continue
            line=line.strip()
            js=json.loads(line)
            data.append(js)
    return data


def save_jsonl(path, data):
    with open(path, 'w') as fp:
        for d in data:
            json.dump(d, fp)
            fp.write('\n')

# +
def load_pkl(path, _range=None):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    if _range is None:
        return data
    else:
        return data[_range[0]:_range[1]]
    
def dump_pkl(path, data):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


# -

def get_idiom_token(i, lang):
    return '<'+str(lang).upper()+'_IDIOM_'+str(i)+'>'


def rm_str(string, keyword):
    return "".join(string.split(keyword))


def get_idiom_path(args, dataset, preprocessing=False, run=False):                    
    idiom_option = 'T:'+str(args.ptsg_threshold)+\
                   '_F:'+str(args.frequency_threshold)+\
                   '_D:'+str(args.depth_threshold)+\
                   '_N:'+str(args.num_of_nodes_threshold)+\
                   '_K:'+str(args.top_k)
    if preprocessing or run:
        # Generating top_k_frags
        # Load code idioms on pretraining / finetuning time
        tmp_path = "/".join(rm_str(args.train_data_file, '../').split(',')).split('/')
    else:
        # Load top_k_frags to generate target dataset idioms
        tmp_path = "/".join(rm_str(args.target_data_file, '../').split(',')).split('/')
    if args.idiom_loss == 'span':
        fn = 'code_idioms.pkl'
    elif args.idiom_loss in ['full', 'none']:
        fn = 'code_idioms_full.pkl'
    else:
        raise NotImplementedError
#     print(args.idiom_path)
#     print([idiom_option, tmp_path[1], args.lang, dataset, 'code_idioms.pkl'])
    if preprocessing:
        return args.idiom_path+'/'.join([idiom_option, tmp_path[1], args.lang])
    elif 'adv_test' in tmp_path:
        return args.idiom_path+'/'.join([idiom_option, tmp_path[1], dataset, fn])
    else:
#         print(args.idiom_path+'/'.join([idiom_option, tmp_path[1], args.lang, dataset, fn]))
#         assert 1 == 2
        return args.idiom_path+'/'.join([idiom_option, tmp_path[1], args.lang, dataset, fn])


def is_none(v):
    if v is None:
        return True
    return False


def is_false(v):
    if type(v) == bool and not v:
        return True
    return False


