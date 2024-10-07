#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_layers import PlanarFlow

class VAEWithFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=50, num_flows=2):
        super(VAEWithFlow, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # For mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Apply sigmoid to ensure output is in range [0, 1]
        )
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_flows)])

    def encode(self, x):
        x = x.view(-1, 784)
        params = self.encoder(x)
        mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        
        log_det_jacobian = 0
        z = z0
        for flow in self.flows:
            z, det_jacobian = flow(z)
            log_det_jacobian += torch.log(det_jacobian + 1e-6)
        
        return z, log_det_jacobian

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, log_det_jacobian

