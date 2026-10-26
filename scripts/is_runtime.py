"""IS-only all-valid write execution; identical ordered FP32 delta updates."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from looplora.sym.writers import FactorizedWriter, OracleWriter

_ORACLE=OracleWriter.forward
_FACTORIZED=FactorizedWriter.forward


def dense_scan(keys,values,lam,beta):
    state=keys.new_zeros(keys.shape[0],keys.shape[-1],keys.shape[-1])
    for t in range(keys.shape[1]):
        key=keys[:,t]
        decayed=lam*state
        error=values[:,t]-torch.bmm(decayed,key.unsqueeze(-1)).squeeze(-1)
        state=decayed+beta*error.unsqueeze(-1)*key.unsqueeze(-2)
    return state


def oracle_forward(self,facts,valid,codes):
    if not bool(valid.all()): return _ORACLE(self,facts,valid,codes)
    rows=torch.arange(len(facts),device=facts.device)[:,None]
    return dense_scan(codes[rows,facts[:,:,0]],codes[rows,facts[:,:,2]],1.,1.)


def factorized_forward(self,x,valid):
    if not bool(valid.all()): return _FACTORIZED(self,x,valid)
    key_u=self.u(x)
    value_u=F.linear(x[...,2*self.d_e:],self.A_v)
    raw=F.linear(key_u,self.K)
    keys=raw/torch.linalg.vector_norm(raw,dim=-1,keepdim=True).clamp_min(1e-6)
    values=F.linear(value_u,self.V)
    return dense_scan(keys,values,self.a_lambda.sigmoid(),self.a_beta.sigmoid()),values


def install():
    OracleWriter.forward=oracle_forward
    FactorizedWriter.forward=factorized_forward
