#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:27:41 2020

@author: wenminggong

TRPO, adapted from https://github.com/HeyuanMingong/irl_cs/blob/master/myrllib/algorithms/trpo.py
"""

import torch
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical, Normal
from torch.optim import Adam, SGD


def detach_distribution(pi):
    # if pi is categorical
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    # if pi is normal
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` policies are valid policies.')
    return distribution


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break
    return x.detach()


class TRPO(object):
    def __init__(self, policy, device='cpu', 
            max_kl=1e-3, cg_iters=10, cg_damping=1e-2, 
            ls_max_steps=10, ls_backtrack_ratio=0.5):
        self.policy = policy
        # self.tau = tau 
        # self.pr_smooth = pr_smooth
        # self.iw_inv = iw_inv
        self.max_kl = max_kl 
        self.cg_iters = cg_iters 
        self.cg_damping = cg_damping
        self.ls_max_steps = ls_max_steps 
        self.ls_backtrack_ratio = ls_backtrack_ratio 
        self.to(device)

    def KL_divergence(self, episodes, old_pi=None):
        # calculate the divergence
        pi = self.policy(episodes.observations)
        # if old_pi is None, then old_pi = pi
        if old_pi is None:
            old_pi = detach_distribution(pi)
        # mask = episodes.mask 
        # if episodes.actions.dim() > 2:
        #     mask = mask.unsqueeze(2)
        # kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
        kl = torch.mean(kl_divergence(pi, old_pi))
        # return mean of different dimensions of the kl_divergence
        return kl 


    def hessian_vector_product(self, episodes, damping=1e-2):
        def _product(vector):
            kl = self.KL_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), 
                    create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            grad2s_copy = []
            for item in grad2s:
                item = item.contiguous(); grad2s_copy.append(item)
            grad2s = tuple(grad2s_copy)
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector 
        return _product 


    def surrogate_loss(self, episodes, old_pi=None):
        with torch.set_grad_enabled(old_pi is None):
            pi = self.policy(episodes.observations)
            if old_pi is None:
                old_pi = detach_distribution(pi)
            
            advantages = episodes.returns 
            log_ratio = pi.log_prob(episodes.actions) - old_pi.log_prob(episodes.actions)
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            # ratio is the importance sampling
            ratio = torch.exp(log_ratio)
            
            loss = - torch.mean(ratio * advantages)
            kl = torch.mean(kl_divergence(pi, old_pi))

        return loss, kl, pi 


    def step(self, episodes):
        max_kl = self.max_kl; cg_iters = self.cg_iters 
        cg_damping = self.cg_damping; ls_max_steps = self.ls_max_steps 
        ls_backtrack_ratio = self.ls_backtrack_ratio 

        old_loss, _, old_pi = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        # convert to vector
        grads = parameters_to_vector(grads)

        # compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, 
                damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, 
                cg_iters=cg_iters)

        # computer the Lagrange multiplier
        shs = 0.5*torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier 

        # save the old parameters 
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size*step, 
                    self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pi=old_pi)
            improve = loss - old_loss 
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break 
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())
    

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.device = device