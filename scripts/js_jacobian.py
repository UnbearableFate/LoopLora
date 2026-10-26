"""Exact final-state ReLU reader Jacobian and differentiable spectral radius."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def jacobian(R,U,Q,memory,z,rho):
    """J(z), including the ReLU gate of the *next* update evaluated at z."""
    h=F.linear(z,R)+F.linear(torch.matmul(memory[:,None],F.linear(z,Q).unsqueeze(-1)).squeeze(-1),U)
    H=R+torch.matmul(torch.matmul(U,memory),Q)
    return rho*(h>0).to(H.dtype)[...,None]*H[:,None]


def spectral_radius(matrix,backend='cpu'):
    original=matrix.device
    if backend=='cpu': matrix=matrix.to('cpu')
    elif backend!='cuda': raise ValueError('spectral backend must be cpu or cuda')
    return torch.linalg.eigvals(matrix).abs().amax(dim=-1).to(original)


def penalty(adapter,prepared,z,backend):
    J=jacobian(adapter.R,adapter.U,adapter.Q,prepared.memory,z,adapter.cfg.rho)
    radius=spectral_radius(J,backend)
    # Every episode has four queries; equal query means followed by episode mean.
    return torch.relu(radius-.9).square().mean(dim=-1).mean(),radius
