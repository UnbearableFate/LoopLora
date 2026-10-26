#!/usr/bin/env python3
"""Choose the exact eigenvalue backend by timing only, before study training."""
from __future__ import annotations
import json
from pathlib import Path
import time
import torch
from run_p1s_f2 import write


def benchmark():
    torch.set_num_threads(1);torch.manual_seed(31415);rows=[]
    base=torch.randn(64,4,32,32,device='cuda')/4
    for backend in ('cpu','cuda'):
        times=[]
        for i in range(7):
            A=base.detach().clone().requires_grad_();torch.cuda.synchronize();started=time.perf_counter()
            from js_jacobian import spectral_radius
            loss=torch.relu(spectral_radius(A,backend)-.9).square().mean();loss.backward();torch.cuda.synchronize()
            if not torch.isfinite(A.grad).all(): raise ValueError('nonfinite exact eigenvalue gradient')
            if i>=2: times.append(time.perf_counter()-started)
        rows.append(dict(backend=backend,seconds_forward_backward=sum(times)/len(times),batch_episodes=64,queries=4,r=32,repeats=5))
    chosen=min(rows,key=lambda r:r['seconds_forward_backward'])['backend']
    out=Path('/work/xg24i002/x10041/LoopLora-wt/symf/reports/js/method.json')
    write(out,dict(method='exact_eigvals',backend=chosen,dtype='float32',selection='minimum measured exact forward/backward time; no scores',timing=rows,
        jacobian='rho * diag(1[Rz+UMQz>0]) * (R+UMQ), evaluated at final state z_L',
        eigenvalue_function='torch.linalg.eigvals(J).abs().amax(-1)',zero_relu_derivative=0,
        gradients='autograd through exact eigenvalues and J parameters/memory; ReLU derivative is locally constant',
        penalty='mean_episode mean_query relu(radius-.9)^2; lambda=1',tests='tests/test_js.py'))
    print(json.dumps(dict(backend=chosen,timing=rows)),flush=True)

if __name__=='__main__': benchmark()
