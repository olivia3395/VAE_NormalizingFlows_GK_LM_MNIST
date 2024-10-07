#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_samples_and_latent_space(model, data_loader, num_samples=16, use_pca=True):
    model.eval()
    latent_dim = model.latent_dim
    
    with torch.no_grad():
        # 生成样本
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.generate(z).cpu()

        # 可视化生成的样本
        plt.figure(figsize=(4, 4))
        grid_img = make_grid(samples.view(num_samples, 1, 28, 28), nrow=4, padding=2)
        plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
        plt.title('Generated Samples')
        plt.show()

    # 潜在空间可视化 (PCA 或 t-SNE)
    if use_pca:
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(latent_space)
    else:
        tsne = TSNE(n_components=2)
        z_2d = tsne.fit_transform(latent_space)
    
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='viridis')
    plt.title('Latent Space Visualization')
    plt.show()

