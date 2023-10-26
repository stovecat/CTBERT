import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import random
from code_idioms.basic_utils import *

def mask_tokens(inputs,tokenizer,args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    unk_mask = labels.eq(tokenizer.unk_token_id)
    probability_matrix.masked_fill_(unk_mask, value=0.0)
        
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def mask_edge(inputs_ids,attn_mask,nodes_mask,args,tokenizer):
    labels=attn_mask.float().clone()
    probability_matrix = torch.full(inputs_ids.shape, args.man_probability).to(inputs_ids.device)
    probability_matrix.masked_fill_(~nodes_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices=masked_indices[:,:,None]&nodes_mask[:,None,:]

    attn_mask[masked_indices]=False
    labels[~masked_indices]=-100
    
    #balance
    probability_matrix = torch.full(labels.shape, labels.eq(1).sum()/(labels.eq(0).sum()+1e-10)).to(inputs_ids.device)
    probability_matrix.masked_fill_(labels.eq(1), value=1.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices]=-100

    return inputs_ids,attn_mask,labels.long()

def mask_align(inputs_ids,attn_mask,nodes_mask,token_mask,args,tokenizer):
    labels=attn_mask.float().clone()
    probability_matrix = torch.full(inputs_ids.shape, args.man_probability).to(inputs_ids.device)
    probability_matrix.masked_fill_(~nodes_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs_ids[masked_indices]=tokenizer.mask_token_id
    masked_indices=masked_indices[:,:,None]&token_mask[:,None,:]

    attn_mask[masked_indices]=False
    attn_mask[masked_indices.transpose(1,2)]=False 
    labels[~masked_indices]=-100
    
    #balance
    probability_matrix = torch.full(labels.shape, labels.eq(1).sum()/(labels.eq(0).sum()+1e-10)).to(inputs_ids.device)
    probability_matrix.masked_fill_(labels.eq(1), value=1.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices]=-100    

    return inputs_ids,attn_mask,labels.long()   

# +
def mask_idiom_span(inputs_ids,attn_mask,idioms_mask,token_mask,args,tokenizer):
    idiom_tokens_indices = idioms_mask.nonzero()
    batch_idx = idiom_tokens_indices[:, 0]
    token_idx = idiom_tokens_indices[:, 1]
    start_end_mask = attn_mask[batch_idx, token_idx, :]
    start_end_idx = (start_end_mask == True).nonzero()
    num_of_idioms = start_end_idx.size()[0]
    start_idx = (start_end_mask == True).nonzero()[[i for i in range(num_of_idioms) if i % 2 == 0], :]
    end_idx = (start_end_mask == True).nonzero()[[i for i in range(num_of_idioms) if i % 2 == 1], :]
    
    def get_span_mask(inputs_ids, attn_mask, batch_idx, token_idx, target_idx, start_end_idx):
        mask = torch.full(attn_mask.shape, 0.).to(inputs_ids.device)
        real_idx = torch.index_select(start_end_idx, 0, torch.tensor([i for i in range(start_end_idx.size()[0]) \
                                                       if i % 2 == 0]).to(start_end_idx.device))[:, 0]
        if not (batch_idx[real_idx].size()[0] == token_idx[real_idx].size()[0] and token_idx[real_idx].size()[0] == target_idx.size()[0]):
            dump_pkl('batch_data.pkl',(inputs_ids,attn_mask,idioms_mask,token_mask,args,tokenizer))
            assert batch_idx[real_idx].size()[0] == token_idx[real_idx].size()[0] and token_idx[real_idx].size()[0] == target_idx.size()[0]   
        else:
            pass
#             dump_pkl('batch_data_normal.pkl',(inputs_ids,attn_mask,idioms_mask,token_mask,args,tokenizer))
        idx = torch.cat([batch_idx[real_idx].view(-1,1), token_idx[real_idx].view(-1,1), target_idx[:, 1].view(-1,1)], dim=1)
        mask[idx.t()[0], idx.t()[1], idx.t()[2]] = True
        return mask
        
    start_labels = get_span_mask(inputs_ids, attn_mask, batch_idx, token_idx, start_idx, start_end_idx)
    end_labels = get_span_mask(inputs_ids, attn_mask, batch_idx, token_idx, end_idx, start_end_idx)
    del idiom_tokens_indices, batch_idx, token_idx, start_end_mask, num_of_idioms, start_idx, end_idx
    torch.cuda.empty_cache() 
    
    
    probability_matrix = torch.full(inputs_ids.shape, args.man_probability).to(inputs_ids.device)
    probability_matrix.masked_fill_(~idioms_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices=masked_indices[:,:,None]&token_mask[:,None,:]

    attn_mask[masked_indices]=False
    attn_mask[masked_indices.transpose(1,2)]=False 
    # start / end indices
    start_labels[~masked_indices]=-100
    end_labels[~masked_indices]=-100

#     print(start_labels.eq(1).sum(), (start_labels.eq(0).sum()+1e-10))

    #balance
    probability_matrix = torch.full(start_labels.shape, start_labels.eq(1).sum()/(start_labels.eq(0).sum()+1e-10)).to(inputs_ids.device)
    probability_matrix.masked_fill_(start_labels.eq(1), value=1.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    start_labels[~masked_indices]=-100
    end_labels[~masked_indices]=-100

#     print(start_labels.eq(1).sum(), start_labels.eq(0).sum())

    return inputs_ids,attn_mask,(start_labels.long(), end_labels.long())


# +
def mask_idiom_full(inputs_ids,attn_mask,idioms_mask,token_mask,args,tokenizer):
#     dump_pkl('mask_idiom_full.pkl', (inputs_ids,attn_mask,idioms_mask,token_mask,args,tokenizer))
    # Predict individual tokens
#     backups = (inputs_ids.clone().cpu(),attn_mask.clone().cpu(),\
#                idioms_mask.clone().cpu(),token_mask.clone().cpu(),args,tokenizer,\
#                inputs_ids.device, attn_mask.device, idioms_mask.device, token_mask.device)
    idiom_tokens_indices = idioms_mask.nonzero()
    batch_idx = idiom_tokens_indices[:, 0]
    token_idx = idiom_tokens_indices[:, 1]
    idiom_mask = attn_mask[batch_idx, token_idx, :]

    labels = torch.full(attn_mask.shape, 0.).to(inputs_ids.device)
    for idx in range(idiom_mask.size()[0]):
        labels[batch_idx[idx], token_idx[idx]] = idiom_mask[idx].clone()

    del idiom_tokens_indices, batch_idx, token_idx, idiom_mask
    torch.cuda.empty_cache() 
    
    probability_matrix = torch.full(inputs_ids.shape, args.man_probability).to(idioms_mask.device)
    probability_matrix.masked_fill_(~idioms_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices=masked_indices[:,:,None]&token_mask[:,None,:]
    attn_mask[masked_indices]=False
    attn_mask[masked_indices.transpose(1,2)]=False 

    labels[~masked_indices]=-100
    
#     backups = [labels.clone().cpu()]
#     backups.append(masked_indices.clone().cpu())
#     backups.append(token_mask.device)
#     bacups = set(backups)

    #balance
    if labels.eq(1).sum()/(labels.eq(0).sum()+1e-10) < 1.:
        probability_matrix = torch.full(labels.shape, labels.eq(1).sum()/(labels.eq(0).sum()+1e-10)).to(inputs_ids.device)
        probability_matrix.masked_fill_(labels.eq(1), value=1.0)
    else:
        probability_matrix = torch.full(labels.shape, labels.eq(0).sum()/(labels.eq(1).sum()+1e-10)).to(inputs_ids.device)
        probability_matrix.masked_fill_(labels.eq(0), value=1.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    labels[~masked_indices]=-100

    return inputs_ids,attn_mask,labels.long()


# -

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(config.hidden_size, config.hidden_size)
        self.qa_outputs2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.qa_outputs3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    def forward(self, inputs_ids, position_idx, attn_mask):
        #MLM for text/code
        inputs_ids,masked_lm_labels=mask_tokens(inputs_ids,self.tokenizer,self.args)
        inputs_masked=masked_lm_labels.ne(-100).float()
        
        #edge or node masked for graph
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(3)
        idioms_mask = position_idx.eq(2)
        rand_val = random.random()
        
        if self.args.not_use_dfg or rand_val < 1/3:
            if self.args.idiom_loss == 'span':
                inputs_ids,attn_mask,edge_labels=mask_idiom_span(inputs_ids,attn_mask,idioms_mask,token_mask,self.args,self.tokenizer)
            elif self.args.idiom_loss == 'full':
                inputs_ids,attn_mask,edge_labels=mask_idiom_full(inputs_ids,attn_mask,idioms_mask,token_mask,self.args,self.tokenizer)
            else:
                raise NotImplementedError
        elif rand_val < 2/3:
            inputs_ids,attn_mask,edge_labels=mask_edge(inputs_ids,attn_mask,nodes_mask,self.args,self.tokenizer)
        else:
            inputs_ids,attn_mask,edge_labels=mask_align(inputs_ids,attn_mask,nodes_mask,token_mask,self.args,self.tokenizer)
            
        #embedding
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        nodes_mask=nodes_mask&inputs_ids.ne(self.tokenizer.mask_token_id)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]

        #forward
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)  
        
        #MLM loss
        sequence_output = outputs[0]
        prediction_scores = self.encoder.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))  
   
        if type(edge_labels) is tuple:
            #idiom loss: span
            start_labels, end_labels = edge_labels
            def get_idiom_loss(sequence_output, labels, linear):
                tmp=torch.tanh(linear(sequence_output))
                score=torch.einsum("abc,adc->abd",tmp.float(),sequence_output.float()).sigmoid()
                scores=torch.cat(((1-score)[:,:,:,None],score[:,:,:,None]),-1)
                return loss_fct(scores.view(-1, 2), labels.view(-1))
            masked_idiom_loss = torch.cat([get_idiom_loss(sequence_output, start_labels, self.qa_outputs2).view(1), \
                                   get_idiom_loss(sequence_output, end_labels, self.qa_outputs3).view(1)]).mean()
            if start_labels.eq(0).sum()!=0:
#                 print(masked_lm_loss,masked_idiom_loss,flush=True)
                return masked_lm_loss, masked_idiom_loss
            else:
                return masked_lm_loss, torch.tensor(0.0)
        else:
            #graph loss
            #idiom loss: full
            tmp=torch.tanh(self.qa_outputs(sequence_output))
            score=torch.einsum("abc,adc->abd",tmp.float(),sequence_output.float()).sigmoid()
            scores=torch.cat(((1-score)[:,:,:,None],score[:,:,:,None]),-1)
            masked_graph_loss=loss_fct(scores.view(-1, 2), edge_labels.view(-1))
            if edge_labels.eq(0).sum()!=0:
                return masked_lm_loss, masked_graph_loss
            else:
                return masked_lm_loss, torch.tensor(0.0)

