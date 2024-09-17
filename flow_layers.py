#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class FlowLayer(nn.Module):
    def __init__(self, latent_dim):
        super(FlowLayer, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u * activation.unsqueeze(-1)

        psi = (1 - torch.tanh(linear).pow(2)).unsqueeze(-1) * self.w
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u.unsqueeze(-1)).squeeze(-1))

        return z_new, det_jacobian


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u * activation.unsqueeze(-1)

        psi = (1 - torch.tanh(linear).pow(2)).unsqueeze(-1) * self.w
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u.unsqueeze(-1)).squeeze(-1))

        return z_new, det_jacobian

