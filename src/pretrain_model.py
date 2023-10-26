import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import random

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

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(config.hidden_size, config.hidden_size)
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    def forward(self, inputs_ids, position_idx, attn_mask):     
        #MLM for text/code
        inputs_ids,masked_lm_labels=mask_tokens(inputs_ids,self.tokenizer,self.args)
        inputs_masked=masked_lm_labels.ne(-100).float()
        
        #edge or node masked for graph
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
        if random.random()>0.5:
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
#         print([list(nd) for nd in position_idx.numpy()])
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)  
    
        #MLM loss
        sequence_output = outputs[0]
        prediction_scores = self.encoder.lm_head(sequence_output)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))  
   
        #graph loss
        tmp=torch.tanh(self.qa_outputs(sequence_output))
        score=torch.einsum("abc,adc->abd",tmp.float(),sequence_output.float()).sigmoid()
        scores=torch.cat(((1-score)[:,:,:,None],score[:,:,:,None]),-1)
        masked_graph_loss=loss_fct(scores.view(-1, 2), edge_labels.view(-1)) 

        if edge_labels.eq(0).sum()!=0:
            return masked_lm_loss, masked_graph_loss
        else:
            return masked_lm_loss, torch.tensor(0.0)

