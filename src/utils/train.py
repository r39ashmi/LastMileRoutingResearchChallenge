
from tqdm import tqdm
import math
import json

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from utils.models.attention_model import set_decode_type
from utils.log_utils import log_values
from utils.functions import move_to
torch.manual_seed(400)

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset,eval_batch_size=1):
    # Validate
    ####### WRITE SEQUENCE LIST
    val_dataloader = DataLoader(dataset, batch_size=eval_batch_size, num_workers=1)

    # Put model in train mode!
    model.eval()
    set_decode_type(model, "greedy")
    
    final_seq={ }
    for batch_id, batch in enumerate(tqdm(val_dataloader, disable=True)):
        stop_seq=batch['stop_seq']
        rt_id=batch['rt_id']
        depot=batch['st_name']
        del batch['stop_seq']
        del batch['rt_id']
        del batch['st_name']
        x = move_to(batch, torch.device("cpu"))
    
        # Evaluate model, get costs and log probabilities
        pi = model(x,distribution='val')
        final_seq[rt_id[0]]={}
        final_seq[rt_id[0]]['proposed']={}
        final_seq[rt_id[0]]['proposed'][depot[0]]=0
        for st_ind in range(len(pi.squeeze())):
            ind=stop_seq[pi.squeeze()[st_ind]-1][0]
            final_seq[rt_id[0]]['proposed'][ind]=st_ind+1 

    with open("model_output_sequence", "w") as fp:
        json.dump(final_seq,fp) 
    

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



def train_batch(
        model,
        optimizer,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,device,grad):

    actual_seq=batch['actual']
    route_score=batch['rt_score'] #changed

    del batch['rt_score'] #changed
    del batch['actual']
    del batch['st_code']
    del batch['rt_id']
    x = move_to(batch, device)
    
    # Evaluate model, get costs and log probabilities
    CE,pi= model(x,actual_seq=actual_seq,distribution='train')
    rt_score_wt=[0.9,0.6,1.0]
    new_CE=rt_score_wt[route_score]*(CE) 
    loss=new_CE 

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, grad)
    optimizer.step()

    # Logging
    if step % int(50) == 0:
        log_values(new_CE, epoch, batch_id)
    return new_CE.item()
