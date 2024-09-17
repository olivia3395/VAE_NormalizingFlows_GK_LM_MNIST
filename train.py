#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from dataloader import get_mnist_dataloaders
from vae_with_flows import VAEWithFlow
from losses import vae_loss_function

def train_vae(model, optimizer, train_loader, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, log_det_jacobian = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_dataloaders()

    model = VAEWithFlow().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_vae(model, optimizer, train_loader, device, epochs=10)

if __name__ == '__main__':
    main()

