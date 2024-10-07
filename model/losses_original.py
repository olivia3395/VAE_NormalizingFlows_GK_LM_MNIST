#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn.functional as F
import torch

def vae_loss_function(recon_x, x, mu, logvar, log_det_jacobian=None):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    if log_det_jacobian is not None:
        KLD = KLD - log_det_jacobian.sum()
    
    return BCE + KLD

