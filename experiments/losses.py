"""Numerically stable device-independent versions of the external recipe losses."""
import torch
from torch.nn import functional as F
from loss.mmd_loss import MMD_loss


def supcon(features, labels, temperature=.1):
    # [batch, views, features], view-major arrangement.
    b,v,_=features.shape
    z=F.normalize(features,dim=-1).transpose(0,1).reshape(b*v,-1)
    y=labels.repeat(v)
    mask=~torch.eye(len(z),dtype=torch.bool,device=z.device)
    positive=(y[:,None]==y[None,:]) & mask
    logits=z@z.T/temperature
    log_prob=logits-logits.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    counts=positive.sum(1)
    valid=counts>0
    if not valid.any(): return features.sum()*0
    return -(log_prob.masked_fill(~positive,0).sum(1)[valid]/counts[valid]).mean()


def lmmd(source,target,labels,prob):
    ws=F.one_hot(labels,7).to(source)
    wt=prob.detach()
    present=(ws.sum(0)>0) & (F.one_hot(wt.argmax(1),7).sum(0)>0)
    if not present.any(): return source.sum()*0
    ws=ws[:,present]/ws[:,present].sum(0).clamp_min(1e-8)
    wt=wt[:,present]/wt[:,present].sum(0).clamp_min(1e-8)
    k=MMD_loss().guassian_kernel(source,target)
    n=len(source)
    return ((ws@ws.T*k[:n,:n]).sum()+(wt@wt.T*k[n:,n:]).sum()-2*(ws@wt.T*k[:n,n:]).sum())/present.sum()


def cdd(p,q):
    matrix=p.T@q
    return matrix.sum()-matrix.trace()


def entropy(logits):
    p=logits.softmax(1)
    return -(p*p.clamp_min(1e-8).log()).sum(1).mean()


def marginal_entropy(logits):
    return -logits.softmax(1).mean(0).clamp_min(1e-6).log().mean()
