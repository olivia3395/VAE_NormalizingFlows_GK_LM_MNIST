#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

def generate_samples(model, num_samples=16, latent_dim=50, device='cpu'):
    model.eval()
    z = torch.randn(num_samples, latent_dim).to(device)
    samples = model.decode(z).cpu()
    
    return samples

