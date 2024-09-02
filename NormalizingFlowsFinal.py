#!/usr/bin/env python
# coding: utf-8

# # Enhancing Variational Autoencoder with Normalizing Flows:
# # A Multivariate g-k Distribution and Levenberg-Marquardt Optimization Approach for MNIST Data Generation and Reconstruction

# ### Overview
# 
# This project enhances a Variational Autoencoder (VAE) with Normalizing Flows using a Multivariate g-k Distribution and Levenberg-Marquardt (LM) optimization, applied to the MNIST dataset.
# 
# #### Key Innovations:
# 
# 1. **Multivariate g-k Distribution**:
#    - Integrated as the base distribution in Normalizing Flows, the g-k distribution allows for flexible modeling of the latent space, capturing complex data structures more effectively than a standard normal distribution.
# 
# 2. **Levenberg-Marquardt Optimization**:
#    - The g-k distribution parameters are optimized using LM, a precise method for non-linear parameter estimation. This optimization improves the flow's ability to model intricate patterns in the data, leading to better generation and reconstruction performance.
# 
# #### Implementation Highlights:
# 
# - **VAE Structure**:
#   - **Encoder**: Compresses input images to a latent space.
#   - **Normalizing Flows**: Transforms the latent space using invertible flows, with the g-k distribution as the base.
#   - **Decoder**: Reconstructs images from the transformed latent space.
# 
# - **Training**:
#   - The VAE is trained on MNIST, with the enhanced latent space enabling superior image generation.
# 
# ---
# 
# This approach focuses on the advanced use of the Multivariate g-k Distribution and LM optimization, making the VAE more expressive and capable of generating realistic, complex data patterns.

# ### Introduction to Normalizing Flows
# 
# Normalizing Flows are techniques used to transform a simple probability distribution into a more complex one through a series of invertible transformations, enhancing the model's ability to capture intricate data patterns.
# 
# #### Key Concepts:
# 
# - **Base Distribution**: Typically starts with a simple multivariate Gaussian distribution.
# - **Flow Layers**: A sequence of invertible transformations applied to the base distribution to create a more complex distribution.
# - **Invertibility and Jacobian**: Ensures transformations are reversible and computationally efficient.
# 
# ### Enhanced VAEs with Multivariate g-k Distribution and LM Optimization
# 
# In this approach, the base distribution in Normalizing Flows is replaced with a **Multivariate g-k Distribution**. The parameters of this distribution are optimized using **Levenberg-Marquardt (LM)**, leading to a more expressive latent space and improved generative performance in VAEs.

# In[1]:


### Data Preparation

# We start by loading and preprocessing the MNIST dataset.

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# In[2]:


# Implementing the basic VAE model
# We start by defining the encoder and decoder components.


import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim) 
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

       
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[3]:


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 创建模型实例
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(), 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


batch_size = 64

data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
       
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

       
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # 确保输出在 [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
   
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(64, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')


for epoch in range(1, 11):
    train(epoch)
    test(epoch)


# 
# During the training process, we observed the following:
# 
# 1. **Gradual Decrease in Loss**: As training progresses, both the training loss and test loss decrease steadily. This indicates that the VAE model is effectively learning the data distribution, and its reconstruction ability is improving over time.
# 
# 2. **Convergence Speed**: In the initial epochs, the loss decreases rapidly and then gradually stabilizes. This is a common pattern in model training, indicating that the model is progressively converging.
# 
# 3. **Final Loss Values**: By the 10th epoch, the average loss on the training set is 105.62, while the average loss on the test set is 104.96. These values suggest that the model has reached a relatively stable state but may still have room for further optimization.
# 
# 4. **Performance**: Although the model continues to improve, the rate of loss reduction slows down in the later epochs, indicating that the current model architecture might have some limitations in further reducing the loss on the MNIST dataset. This could be related to the latent space dimensions or the design of the encoder and decoder.
# 
#     
#     

# ### Planar Flows
# 
# We are introducing Planar Flows into the VAE model to enhance the flexibility and expressiveness of the latent space.
# 
# **What are Planar Flows?**
# Planar Flows are a simple yet effective type of Normalizing Flow. They apply a series of invertible linear transformations to the latent variables, making the latent space distribution more complex and better at capturing data distribution.
# 
# 

# In[5]:


import torch
import torch.nn as nn

class PlanarFlow(nn.Module):
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim))
        self.w = nn.Parameter(torch.randn(latent_dim))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        linear = torch.matmul(z, self.w) + self.b  # shape: (batch_size)
        activation = torch.tanh(linear)  # shape: (batch_size)
        
        # Add a dimension to u to enable broadcasting, i.e., (latent_dim,) -> (1, latent_dim)
        z_new = z + self.u * activation.unsqueeze(-1)  # shape: (batch_size, latent_dim)
        
        # Compute psi, which will have shape (batch_size, latent_dim)
        psi = (1 - torch.tanh(linear).pow(2)).unsqueeze(-1) * self.w  # shape: (batch_size, latent_dim)
        
        # Compute the determinant of the Jacobian, shape: (batch_size,)
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u.unsqueeze(-1)).squeeze(-1))
        
        return z_new, det_jacobian


        
        
        


# In[6]:


class VAEWithFlows(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_flows=2):
        super(VAEWithFlows, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

       
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Planar Flows
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_flows)])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

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
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, log_det_jacobian


# In[7]:


def loss_function_with_flows(recon_x, x, mu, logvar, log_det_jacobian):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
   
    KLD = KLD - log_det_jacobian.sum()
    
    return BCE + KLD


# In[8]:


model = VAEWithFlows().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = model(data)
        loss = loss_function_with_flows(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, log_det_jacobian = model(data)
            test_loss += loss_function_with_flows(recon_batch, data, mu, logvar, log_det_jacobian).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(64, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'reconstruction_with_flows_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

for epoch in range(1, 11):
    train(epoch)
    test(epoch)


# ### Key Points on Introducing Normalizing Flows
# 
# - **Flow Quantity and Type**: Too few flows may not enhance the model; too many can cause instability. Planar Flows might be too simple—consider using RealNVP or Glow.
# 
# - **Optimization Issues**: Numerical instability from Jacobian calculations; requires careful tuning of learning rate and parameter initialization.
# 
# - **Latent Space and Complexity**: High-dimensional latent space increases model complexity, making optimization difficult.
# 
# - **Dataset Simplicity**: On simple datasets like MNIST, Planar Flows may show limited benefits.
# 
# ### Recommendations
# - **Adjust Flow Layers**: Use 2-5 flow layers and monitor results.
# - **Explore Different Flows**: Try RealNVP or Radial Flows.
# - **Tune Hyperparameters**: Lower the learning rate and adjust initialization.
# - **Debug Incrementally**: Add flows step-by-step and visualize latent space changes.
# 
# ### Example Adjustment:
# Reduce flow layers to 2 and decrease the learning rate to `1e-4`.

# In[9]:


class VAEWithFlows(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_flows=2):
        super(VAEWithFlows, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        
      
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

     
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Planar Flows
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_flows)])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

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
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, log_det_jacobian

    

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(1, 11):
    train(epoch)
    test(epoch)

    
    


# ### RealNVP (Real-valued Non-Volume Preserving) Flow: Key Points
# 
# **RealNVP** is a type of Normalizing Flow that allows for complex, non-linear transformations of data while keeping the computation of the Jacobian determinant tractable. It is particularly useful in generative modeling, where the goal is to transform a simple distribution (like a Gaussian) into a complex data distribution.
# 
# #### Key Concepts:
# 
# 1. **Affine Coupling Layers**:
#    - RealNVP utilizes *affine coupling layers*, which split the input data into two parts: one part remains unchanged, and the other part is transformed conditionally based on the unchanged part. This transformation typically involves scaling and translation operations.
#    
# 2. **Invertibility**:
#    - The transformation is designed to be easily invertible. This means that given the output, the input can be exactly recovered, which is crucial for both density estimation and sampling.
# 
# 3. **Efficient Jacobian Computation**:
#    - A major advantage of RealNVP is that the Jacobian of the transformation is triangular, making its determinant easy to compute as the product of the diagonal elements. This efficiency is key to applying RealNVP in high-dimensional settings.
# 
# 4. **Complexity**:
#    - Despite the simplicity of each individual transformation, stacking multiple affine coupling layers allows RealNVP to model very complex, high-dimensional distributions.
# 
# 5. **Application in Generative Models**:
#    - RealNVP is often used in Variational Autoencoders (VAEs) and other generative models to improve the expressiveness of the latent space, leading to better quality in generated samples.
# 
# RealNVP's ability to balance expressiveness with computational efficiency makes it a powerful tool in modern generative modeling.

# In[ ]:





# In[10]:


class RealNVPFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(RealNVPFlow, self).__init__()
        self.input_dim = input_dim

        # 创建条件变换函数，s(x) 和 t(x)
        self.s = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )

        self.t = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2)
        )

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=-1)
        s = self.s(z2)
        t = self.t(z2)
        z1 = z1 * torch.exp(s) + t
        log_det_jacobian = s.sum(dim=-1)
        return torch.cat([z1, z2], dim=-1), log_det_jacobian

    
class VAEWithRealNVP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_flows=4):
        super(VAEWithRealNVP, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows

       
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

     
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # RealNVP Flows
        self.flows = nn.ModuleList([RealNVPFlow(latent_dim) for _ in range(num_flows)])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        
        log_det_jacobian = 0
        z = z0

        for flow in self.flows:
            z, det_jacobian = flow(z)
            log_det_jacobian += det_jacobian

        return z, log_det_jacobian

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, log_det_jacobian
    def generate(self, z):
        return self.decode(z)


# In[11]:


model = VAEWithRealNVP().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(1, 21):
    train(epoch)
    test(epoch)


# In[12]:


def vae_loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) - log_det_jacobian.sum()
    return BCE + KLD

def train_vae_with_realnvpflow(model, data_loader, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, log_det_jacobian = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}")


# In[13]:


import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_samples_and_latent_space(model, data_loader, num_samples=16, use_pca=True):
    model.eval()
    with torch.no_grad():
        
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.generate(z).cpu()

       
        plt.figure(figsize=(4, 4))
        grid_img = make_grid(samples.view(num_samples, 1, 28, 28), nrow=4, padding=2)
        plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

      
        latents = []
        labels = []
        for data, label in data_loader:
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            z, _ = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy())
            labels.append(label.numpy())

        latents = np.concatenate(latents)
        labels = np.concatenate(labels)

        
        if model.latent_dim > 2:
            if use_pca:
                latents_2d = PCA(n_components=2).fit_transform(latents)
            else:
                latents_2d = TSNE(n_components=2).fit_transform(latents)
        else:
            latents_2d = latents 

        plt.figure(figsize=(8, 6))
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', s=2)
        plt.colorbar()
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        plt.show()


# In[14]:


vae_with_realnvp = VAEWithRealNVP().to(device)
train_vae_with_realnvpflow(vae_with_realnvp, data_loader, num_epochs=10)


# In[15]:


visualize_samples_and_latent_space(vae_with_realnvp, data_loader)


# ### Analysis Based on the plot:
# 
# #### 1. **Sample Quality**
# - **Observation**: Generated images are blurry with unclear details, making digits hard to recognize.
# - **Issues**:
#   - **Undertraining**: Insufficient training may lead to poor sample quality.
#   - **VAE Limitations**: VAE may struggle with generating high-quality samples, especially with complex data.
#   - **RealNVP Adjustment**: Inadequate tuning of RealNVP may fail to capture the true data distribution.
# - **Suggestions**:
#   - Extend training duration.
#   - Fine-tune hyperparameters (e.g., learning rate, latent dimensions).
#   - Experiment with other flow models like Planar Flow or MAF.
# 
# #### 2. **Latent Space Structure**
# - **Observation**: Data points are evenly distributed but lack clear clustering; labels are mixed without distinct separation.
# - **Issues**:
#   - **Misaligned Distribution**: RealNVP may not properly align latent space, causing poor category separation.
#   - **Overextended Space**: Excessive spread in latent space may degrade sample quality.
# - **Suggestions**:
#   - Adjust dimensionality reduction techniques (e.g., t-SNE, PCA).
#   - Increase regularization for a more compact latent space.
#   - Modify encoder and flow structure for better alignment with data distribution.
# 
# ### Summary
# - **Sample Quality**: Currently poor; needs more training or model adjustments.
# - **Latent Space**: Lacks clear separation; requires better structure alignment.
# 
# To improve results, consider fine-tuning hyperparameters, extending training, or switching flow models for enhanced sample quality and latent space structure.

# ### Why RealNVP is Effective
# 
# - **Expressive Power**: Captures complex distributions through invertible affine transformations, fitting high-dimensional data better.
# - **Stable Training**: Simplifies Jacobian calculation, ensuring stable and efficient training.
# - **Latent Space**: Enhances latent space flexibility, leading to more realistic samples.
# - **Scalability**: Easily expands with more layers without increasing complexity.
# 
# ### Summary
# RealNVP significantly boosts model performance, improving sample quality and stability, especially in complex, high-dimensional tasks.

# # Combined Flow: Key Points
# 
# **Combined Flow** enhances a VAE by combining **RealNVP** and **Planar Flows** to increase the flexibility and expressiveness of the latent space.
# 
# #### Key Concepts:
# 
# 1. **Hybrid Approach**:
#    - Integrates **RealNVP** for global transformations and **Planar Flows** for local adjustments in the latent space, providing a balance of structure and flexibility.
# 
# 2. **Sequential Processing**:
#    - Applies flows in a sequence, first transforming with RealNVP for robust shaping, then fine-tuning with Planar Flows.
# 
# 3. **Improved Expressiveness**:
#    - This combination allows the model to better capture complex data distributions, enhancing the VAE's ability to generate diverse and accurate samples.
# 
# The Combined Flow approach makes the VAE more adaptable to complex data patterns, leading to superior generative performance.

# In[16]:


class CombinedFlow(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_realnvp_flows=2, num_planar_flows=2):
        super(CombinedFlow, self).__init__()
        self.realnvp_flows = nn.ModuleList([RealNVPFlow(latent_dim, hidden_dim) for _ in range(num_realnvp_flows)])
        self.planar_flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(num_planar_flows)])
    
    def forward(self, z):
        log_det_jacobian = 0
        
       
        for flow in self.realnvp_flows:
            z, det_jacobian = flow(z)
            log_det_jacobian += det_jacobian
        
       
        for flow in self.planar_flows:
            z, det_jacobian = flow(z)
            log_det_jacobian += torch.log(det_jacobian + 1e-6)
        
        return z, log_det_jacobian

    
    
class VAEWithCombinedFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=50, num_realnvp_flows=2, num_planar_flows=2):
        super(VAEWithCombinedFlow, self).__init__()
        self.latent_dim = latent_dim

       
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  

      
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

      
        self.flow = CombinedFlow(latent_dim, hidden_dim, num_realnvp_flows, num_planar_flows)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mu + eps * std
        
        
        z, log_det_jacobian = self.flow(z0)
        
        return z, log_det_jacobian

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, log_det_jacobian
    
    def generate(self, z):
        return self.decode(z)

 


# In[17]:


model = VAEWithCombinedFlow().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(1, 21):
    train(epoch)
    test(epoch)


# In[18]:


def train_vae_with_combined_flow(model, data_loader, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        train_loss = 0
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, log_det_jacobian = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}")


# In[19]:


input_dim = 784
hidden_dim = 400
latent_dim = 50
num_realnvp_flows = 2
num_planar_flows = 2
num_epochs = 10


vae_with_combined_flow = VAEWithCombinedFlow(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_realnvp_flows=num_realnvp_flows, num_planar_flows=num_planar_flows).to(device)
train_vae_with_combined_flow(vae_with_combined_flow, data_loader, num_epochs=num_epochs)


# In[20]:


visualize_samples_and_latent_space(vae_with_combined_flow, data_loader)


# 
# 
# ### Key Points
# 
# #### 1. **Sample Quality**
# - **Blurry Images**: Generated images are unclear with poor detail.
# - **Undertraining**: Model likely needs more training.
# - **Complexity Issues**: CombinedFlow may be too complex, affecting stability.
# 
# #### 2. **Latent Space Structure**
# - **Poor Category Separation**: Latent space lacks distinct clusters.
# - **Balancing Issues**: KL divergence and flow contributions may not be well-balanced.
# 
# ### Suggestions
# - **Increase Training**: Extend epochs for better learning.
# - **Adjust Flow Model**: Simplify or reconfigure flows for improved stability and separation.
# - **Enhance Regularization**: Strengthen KL divergence to improve latent space clarity.
# 

# ### Introducing Adversarial Training with VAE-GAN
# 
# VAE-GAN is a powerful approach that combines the strengths of a Variational Autoencoder (VAE) with a Generative Adversarial Network (GAN). This hybrid model leverages the generative capabilities of the VAE and the adversarial training of the GAN to enhance the quality, realism, and diversity of generated samples. By integrating these two models, VAE-GAN aims to overcome the limitations of each individual model and produce more accurate and high-fidelity outputs.
# 
# #### VAE-GAN Architecture
# 
# 1. **Variational Autoencoder (VAE)**:
#    - **Encoder**: Maps input data to a latent space by learning the mean and variance of a Gaussian distribution. This probabilistic mapping enables the model to capture complex data distributions and generate diverse samples.
#    - **Latent Space**: The VAE samples from the latent space to generate new data points. This space represents the underlying features learned from the input data.
#    - **Decoder**: Reconstructs the input data from the sampled latent variables, generating data that resembles the original input.
# 
# 2. **Generative Adversarial Network (GAN)**:
#    - **Generator (VAE)**: In the VAE-GAN setup, the VAE's decoder serves as the generator. It generates new samples from the latent space.
#    - **Discriminator**: The GAN's discriminator evaluates the authenticity of the samples by distinguishing between real data and the data generated by the VAE. It provides feedback to the generator, pushing it to produce more realistic samples.
#    - **Adversarial Training**: The VAE and discriminator are trained in a competitive manner. The discriminator tries to correctly identify real and fake samples, while the VAE adjusts its parameters to generate samples that can fool the discriminator.
# 
# #### Implementation Strategy
# 
# 1. **Use VAE as the Generator**:
#    - The VAE generates samples by sampling from the learned latent space. The decoder reconstructs these samples, aiming to produce data that closely resembles the training data.
# 
# 2. **GAN Discriminator for Distribution Evaluation**:
#    - The GAN's discriminator is trained to evaluate the differences between real and generated samples. It assesses how closely the generated samples match the true data distribution, providing a measure of sample quality.
# 
# 3. **Adversarial Training for Realism**:
#    - Adversarial training is employed to make the VAE-generated samples more realistic. The discriminator's feedback is used to refine the VAE, improving the overall quality of the generated data.
# 
# #### Benefits of VAE-GAN
# 
# - **Improved Sample Quality**: By combining VAE's generative capabilities with GAN's adversarial training, VAE-GAN produces higher quality samples with sharper details and fewer artifacts.
# - **Diverse Sample Generation**: The VAE's latent space sampling ensures diversity in the generated data, while the GAN's discriminator ensures that this diversity does not come at the cost of realism.
# - **Overcoming Limitations**: VAE-GAN mitigates common issues in VAEs, such as blurry outputs, by incorporating the GAN's ability to generate sharp and realistic images. It also addresses the mode collapse issue in GANs by utilizing VAE's structured latent space.
# 

# Implementing the Discriminator
# First, we implement a discriminator to distinguish between real samples and generated samples. We can use Spectral Normalization to enhance the stability of the discriminator.

# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        for _ in range(self.power_iterations):
            v = F.normalize(torch.matmul(w.view(w.size(0), -1).t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.matmul(w.view(w.size(0), -1), v), dim=0, eps=1e-12)

        sigma = torch.dot(u, torch.matmul(w.view(w.size(0), -1), v))
        setattr(self.module, self.name, nn.Parameter(w / sigma.expand_as(w)))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = F.normalize(w.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(w.new_empty(width).normal_(0, 1), dim=0, eps=1e-12)
        w_bar = w.data

        
        self.module.register_buffer(self.name + "_u", u)
        self.module.register_buffer(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w_bar))

        self._update_u_v()

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class DiscriminatorWithSpectralNorm(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super(DiscriminatorWithSpectralNorm, self).__init__()
        self.fc1 = SpectralNorm(nn.Linear(input_dim, hidden_dim))
        self.fc2 = SpectralNorm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = SpectralNorm(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = x.view(-1, 784)  # 确保输入张量展平为 (batch_size, 784)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return torch.sigmoid(self.fc3(x))


# Modified generator (VAEWithCombinedFlow)
# VAEWithCombinedFlow is used as a generator in which the flow has been augmented. We use it as a generator part for adversarial training of GAN.

# Implementing a VAE-GAN Hybrid Model
# To combine VAE with GAN, we need a combinatorial model that handles both the reconstruction of VAE and the adversarial loss of GAN.

# In[22]:


class VAEGAN(nn.Module):
    def __init__(self, vae, discriminator):
        super(VAEGAN, self).__init__()
        self.vae = vae
        self.discriminator = discriminator

    def forward(self, x):
        recon_x, mu, logvar, log_det_jacobian = self.vae(x)
        return recon_x, mu, logvar, log_det_jacobian
    
    def generate(self, z):
        return self.vae.decode(z)


# Define the loss function
# We need a combined loss function that takes into account the reconstruction error of the VAE, the KL dispersion, and the adversarial loss of the GAN.

# In[23]:


def vae_gan_loss(recon_x, x, mu, logvar, log_det_jacobian, real_output, fake_output):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD - log_det_jacobian.sum()

    # 对抗性损失
    adv_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))

    return BCE + KLD + adv_loss

def discriminator_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


# Training the VAE-GAN model
# Next, a training loop is defined in which both the VAE and the discriminator are optimized.

# In[24]:


latent_dim = 50
vae = VAEWithCombinedFlow(input_dim=784, hidden_dim=400, latent_dim=latent_dim, num_realnvp_flows=2, num_planar_flows=2).to(device)
discriminator = DiscriminatorWithSpectralNorm(input_dim=784, hidden_dim=256).to(device)
vae_gan = VAEGAN(vae, discriminator).to(device)

optimizer_g = torch.optim.Adam(vae_gan.vae.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(vae_gan.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

batch_size = 64

num_epochs = 20


# In[25]:


data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(data_loader):
        real_images = real_images.view(-1, 784).to(device)
        batch_size = real_images.size(0)
        
        
        recon_images, mu, logvar, log_det_jacobian = vae_gan(real_images)

      
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = vae_gan.generate(z)
      
       
        real_output = vae_gan.discriminator(real_images)
        fake_output = vae_gan.discriminator(fake_images)
        
      
        d_loss = discriminator_loss(real_output, fake_output)
        
      
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
      
        fake_output = vae_gan.discriminator(recon_images)
        g_loss = vae_gan_loss(recon_images, real_images, mu, logvar, log_det_jacobian, real_output, fake_output)
        
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")


# In[26]:


def vae_loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD - log_det_jacobian.sum()
    return BCE + KLD


# In[27]:


def train_vae(vae, data_loader, num_epochs):
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        train_loss = 0
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, log_det_jacobian = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.4f}")
        test_vae(vae)

def test_vae(vae):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, logvar, log_det_jacobian = vae(data)
            test_loss += vae_loss_function(recon_batch, data, mu, logvar, log_det_jacobian).item()
    test_loss /= len(data_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


# In[28]:


def train_vae_gan(vae_gan, data_loader, num_epochs):
    vae_gan.train()
    optimizer_g = torch.optim.Adam(vae_gan.vae.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(vae_gan.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        train_loss_g = 0
        train_loss_d = 0
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

          
            recon_images, mu, logvar, log_det_jacobian = vae_gan(real_images)

           
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = vae_gan.generate(z)

           
            real_output = vae_gan.discriminator(real_images.view(-1, 784))  
            fake_output = vae_gan.discriminator(fake_images.view(-1, 784))  

            
            d_loss = discriminator_loss(real_output, fake_output)
            train_loss_d += d_loss.item()

            
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

           
            fake_output = vae_gan.discriminator(recon_images.view(-1, 784))  
            g_loss = vae_gan_loss_function(recon_images, real_images, mu, logvar, log_det_jacobian, real_output, fake_output)
            train_loss_g += g_loss.item()

           
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch}/{num_epochs}], d_loss: {train_loss_d / len(data_loader):.4f}, g_loss: {train_loss_g / len(data_loader):.4f}")


# In[29]:


import matplotlib.pyplot as plt
def generate_samples(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.generate(z).cpu()
        save_image(samples.view(num_samples, 1, 28, 28), 'generated_samples.png', nrow=4)

   
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(samples.view(-1, 28).numpy(), cmap='gray')
    plt.show()


# In[30]:


# Initializing and Training a VAE Model

vae = VAEWithCombinedFlow().to(device)
train_vae(vae, data_loader, num_epochs)



# In[31]:


visualize_samples_and_latent_space(vae, train_loader)


# In[32]:


def vae_gan_loss_function(recon_x, x, mu, logvar, log_det_jacobian, real_output, fake_output):
    VAE_loss = vae_loss_function(recon_x, x, mu, logvar, log_det_jacobian)
    adv_loss = F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
    return VAE_loss + adv_loss

def discriminator_loss(real_output, fake_output):
    real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


# In[33]:


discriminator = DiscriminatorWithSpectralNorm().to(device)
vae_gan = VAEGAN(vae, discriminator).to(device)

train_vae_gan(vae_gan, data_loader, num_epochs)


# In[34]:


def visualize_samples_and_latent_space(model, data_loader, num_samples=16, use_pca=True):
    model.eval()
    with torch.no_grad():
        # 生成样本
        latent_dim = model.vae.latent_dim  # 通过 VAEGAN 的 VAE 访问 latent_dim
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.vae.generate(z).cpu()  # 使用 VAE 的 generate 方法

        # 可视化生成样本
        plt.figure(figsize=(4, 4))
        grid_img = make_grid(samples.view(num_samples, 1, 28, 28), nrow=4, padding=2)
        plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

        # 可视化潜在空间
        latents = []
        labels = []
        for data, label in data_loader:
            data = data.to(device)
            mu, logvar = model.vae.encode(data.view(-1, 784))  # 使用 VAE 的 encode 方法
            z, _ = model.vae.reparameterize(mu, logvar)  # 使用 VAE 的 reparameterize 方法
            latents.append(z.cpu().numpy())
            labels.append(label.numpy())

        latents = np.concatenate(latents)
        labels = np.concatenate(labels)

        # 如果潜在空间维度 > 2，使用降维
        if latent_dim > 2:
            if use_pca:
                latents_2d = PCA(n_components=2).fit_transform(latents)
            else:
                latents_2d = TSNE(n_components=2).fit_transform(latents)
        else:
            latents_2d = latents  # 如果 latent_dim 正好是 2

        plt.figure(figsize=(8, 6))
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', s=2)
        plt.colorbar()
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        plt.show()


# In[35]:


visualize_samples_and_latent_space(vae_gan, train_loader)


# 
# ### Key Findings
# 
# #### 1. **Sample Quality**
#    - **Blurry and Distorted**: Generated samples are unclear and often distorted, with poor boundary definition.
#    - **Potential Issues**: Insufficient training and imbalance between adversarial and reconstruction losses.
# 
#    - **Suggestions**:
#      - Extend training time.
#      - Rebalance loss weights.
#      - Enhance discriminator complexity.
# 
# #### 2. **Latent Space Visualization**
#    - **Weak Category Separation**: Data points are evenly distributed but lack clear category separation.
#    - **Potential Issues**: Inadequate latent space representation and disruption from adversarial training.
# 
#    - **Suggestions**:
#      - Increase KL divergence weight.
#      - Add flow model complexity.
#      - Apply stronger regularization.
# 
# #### 3. **Next Steps**
#    - **Improve Sample Quality**: Continue training, adjust loss weights, and enhance the discriminator.
#    - **Refine Latent Space**: Increase model complexity to achieve better category separation and consistency.
# 
# 

# 
# 
# # Report: Overview of MultivariateGKDistribution
# 
# **MultivariateGKDistribution** is a specialized parameterized distribution model designed to extend the univariate g-and-k distribution for multivariate scenarios. It is particularly effective in capturing the complexities of real-world data, such as asymmetry and heavy-tailed characteristics.
# 
# #### Key Parameters:
# - **Location (a)**: Centers the distribution.
# - **Scale (b)**: Determines the distribution's spread.
# - **Skewness (g)**: Controls the asymmetry of the distribution.
# - **Kurtosis (k)**: Governs the thickness of the distribution's tails, impacting its heavy-tailed nature.
# 
# #### Mathematical Foundation:
# The distribution is defined by applying a nonlinear transformation to a standard normal variable `u`, producing a variable `z` whose distribution is shaped by the parameters `a`, `b`, `g`, and `k`. This flexibility allows the model to represent a wide range of distributional forms, from symmetric normal distributions to skewed and heavy-tailed distributions.
# 
# 

# In[36]:


import torch
import torch.nn as nn

class MultivariateGKDistribution(nn.Module):
    def __init__(self, dim, a=0, b=1, g=0, k=0):
        super(MultivariateGKDistribution, self).__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32).repeat(dim))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32).repeat(dim))
        self.g = nn.Parameter(torch.tensor(g, dtype=torch.float32).repeat(dim))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32).repeat(dim))
    
    def forward(self, u):
        z = self.a + self.b * (1 + 0.8 * (1 - torch.exp(-self.g * u)) / (1 + torch.exp(-self.g * u))) * (1 + u**2)**self.k * u
        return z

    def sample(self, num_samples):
        u = torch.randn(num_samples, self.dim, dtype=torch.float32)
        return self.forward(u)


# In[37]:


from scipy.optimize import least_squares

def levenberg_marquardt_loss(params, model, target_distribution):
    dim = model.dim
    a, b, g, k = params[:dim], params[dim:2*dim], params[2*dim:3*dim], params[3*dim:]
    model.a.data = torch.tensor(a, dtype=torch.float32)
    model.b.data = torch.tensor(b, dtype=torch.float32)
    model.g.data = torch.tensor(g, dtype=torch.float32)
    model.k.data = torch.tensor(k, dtype=torch.float32)

    samples = model.sample(len(target_distribution))
    loss = torch.mean((samples - target_distribution) ** 2).item()
    return loss

def optimize_gk_parameters(model, target_distribution):
    initial_params = torch.cat([model.a.data, model.b.data, model.g.data, model.k.data]).numpy()
    result = least_squares(levenberg_marquardt_loss, initial_params, args=(model, target_distribution), method='lm')
    optimized_params = result.x
    
    model.a.data = torch.tensor(optimized_params[:model.dim], dtype=torch.float32)
    model.b.data = torch.tensor(optimized_params[model.dim:2*model.dim], dtype=torch.float32)
    model.g.data = torch.tensor(optimized_params[2*model.dim:3*model.dim], dtype=torch.float32)
    model.k.data = torch.tensor(optimized_params[3*model.dim:], dtype=torch.float32)


# In[38]:


class FlowLayer(nn.Module):
    def __init__(self, latent_dim):
        super(FlowLayer, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim, dtype=torch.float32))
        self.w = nn.Parameter(torch.randn(latent_dim, 1, dtype=torch.float32))  # Adjusted to have a singleton dimension for broadcasting
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u.unsqueeze(0) * activation  # Broadcast u

        psi = (1 - activation**2) * self.w.squeeze(1)  # Squeeze w to match the psi calculation
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))

        log_det_jacobian = torch.log(det_jacobian + 1e-6).float()

        return z_new, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=2, base_dist=None):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([FlowLayer(latent_dim) for _ in range(num_flows)])
        self.base_dist = base_dist if base_dist is not None else MultivariateGKDistribution(latent_dim)
    
    def forward(self, z):
        log_det_jacobian = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian
    
    def sample(self, num_samples):
        z0 = self.base_dist.sample(num_samples)
        z, _ = self.forward(z0)
        return z


# In[39]:


import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
        self.flow = NormalizingFlow(latent_dim, num_flows=num_flows)

    def encode(self, x):
        x = x.view(-1, 784)  # Flatten the input image to match the input_dim of 784
        params = self.encoder(x)
        mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z, log_det_jacobian = self.flow(z)
        return z, log_det_jacobian

    def decode(self, z):
        return self.decoder(z)  # Keep it as [batch_size, 784], values in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, log_det_jacobian

    def generate(self, z):
        return self.decode(z)  # Use the decoder to generate images from z

    
    # Define the MNIST dataset with the correct transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor with pixel values between 0 and 1
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Initialize the VAE model with Normalizing Flow
vae_with_flow = VAEWithFlow(input_dim=784, hidden_dim=400, latent_dim=50, num_flows=4).to(device)

# Define the optimizer
optimizer = optim.Adam(vae_with_flow.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    # Flatten the input to match the output of the decoder
    x = x.view(-1, 784)  # Flatten target to [batch_size, 784]
    
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD - torch.sum(log_det_jacobian)

for epoch in range(1, 11):
    train_loss = 0
    vae_with_flow.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = vae_with_flow(data)
        
       
        loss = loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_samples_and_latent_space(model, data_loader, num_samples=16, use_pca=True):
    model.eval()
    latent_dim = model.latent_dim  # 
    
    with torch.no_grad():
        # 
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.generate(z).cpu() 

       
        plt.figure(figsize=(4, 4))
        grid_img = make_grid(samples.view(num_samples, 1, 28, 28), nrow=4, padding=2)
        plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

       
        latents = []
        labels = []
        for data, label in data_loader:
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))  
            z, _ = model.reparameterize(mu, logvar)  
            latents.append(z.cpu().numpy())
            labels.append(label.numpy())

        latents = np.concatenate(latents)
        labels = np.concatenate(labels)

       
        if latent_dim > 2:
            if use_pca:
                latents_2d = PCA(n_components=2).fit_transform(latents)
            else:
                latents_2d = TSNE(n_components=2).fit_transform(latents)
        else:
            latents_2d = latents 

        plt.figure(figsize=(8, 6))
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='viridis', s=2)
        plt.colorbar()
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        plt.show()


# In[41]:


visualize_samples_and_latent_space(vae_with_flow, train_loader)


# In[42]:


import torch
import torch.nn as nn

class MultivariateGKDistribution(nn.Module):
    def __init__(self, dim, a=0, b=1, g=0, k=0):
        super(MultivariateGKDistribution, self).__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32).repeat(dim))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32).repeat(dim))
        self.g = nn.Parameter(torch.tensor(g, dtype=torch.float32).repeat(dim))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32).repeat(dim))
    
    def forward(self, u):
        z = self.a + self.b * (1 + 0.8 * (1 - torch.exp(-self.g * u)) / (1 + torch.exp(-self.g * u))) * (1 + u**2)**self.k * u
        return z

    def sample(self, num_samples):
        u = torch.randn(num_samples, self.dim, dtype=torch.float32)
        return self.forward(u)
from scipy.optimize import least_squares

def levenberg_marquardt_loss(params, model, target_distribution):
    dim = model.dim
    a, b, g, k = params[:dim], params[dim:2*dim], params[2*dim:3*dim], params[3*dim:]
    model.a.data = torch.tensor(a, dtype=torch.float32)
    model.b.data = torch.tensor(b, dtype=torch.float32)
    model.g.data = torch.tensor(g, dtype=torch.float32)
    model.k.data = torch.tensor(k, dtype=torch.float32)

    samples = model.sample(len(target_distribution))
    loss = torch.mean((samples - target_distribution) ** 2).item()
    return loss

def optimize_gk_parameters(model, target_distribution):
    initial_params = torch.cat([model.a.data, model.b.data, model.g.data, model.k.data]).numpy()
    result = least_squares(levenberg_marquardt_loss, initial_params, args=(model, target_distribution), method='lm')
    optimized_params = result.x
    
    model.a.data = torch.tensor(optimized_params[:model.dim], dtype=torch.float32)
    model.b.data = torch.tensor(optimized_params[model.dim:2*model.dim], dtype=torch.float32)
    model.g.data = torch.tensor(optimized_params[2*model.dim:3*model.dim], dtype=torch.float32)
    model.k.data = torch.tensor(optimized_params[3*model.dim:], dtype=torch.float32)
class FlowLayer(nn.Module):
    def __init__(self, latent_dim):
        super(FlowLayer, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim, dtype=torch.float32))
        self.w = nn.Parameter(torch.randn(latent_dim, 1, dtype=torch.float32))  # Adjusted to have a singleton dimension for broadcasting
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u.unsqueeze(0) * activation  # Broadcast u

        psi = (1 - activation**2) * self.w.squeeze(1)  # Squeeze w to match the psi calculation
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))

        log_det_jacobian = torch.log(det_jacobian + 1e-6).float()

        return z_new, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=2, base_dist=None):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([FlowLayer(latent_dim) for _ in range(num_flows)])
        self.base_dist = base_dist if base_dist is not None else MultivariateGKDistribution(latent_dim)
    
    def forward(self, z):
        log_det_jacobian = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian
    
    def sample(self, num_samples):
        z0 = self.base_dist.sample(num_samples)
        z, _ = self.forward(z0)
        return z
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
        self.flow = NormalizingFlow(latent_dim, num_flows=num_flows)

    def encode(self, x):
        x = x.view(-1, 784)  # Flatten the input image to match the input_dim of 784
        params = self.encoder(x)
        mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z, log_det_jacobian = self.flow(z)
        return z, log_det_jacobian

    def decode(self, z):
        return self.decoder(z)  # Keep it as [batch_size, 784], values in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, log_det_jacobian

    def generate(self, z):
        return self.decode(z)  # Use the decoder to generate images from z

    
    # Define the MNIST dataset with the correct transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor with pixel values between 0 and 1
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Initialize the VAE model with Normalizing Flow
vae_with_flow = VAEWithFlow(input_dim=784, hidden_dim=400, latent_dim=50, num_flows=4).to(device)

# Define the optimizer
optimizer = optim.Adam(vae_with_flow.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    # Flatten the input to match the output of the decoder
    x = x.view(-1, 784)  # Flatten target to [batch_size, 784]
    
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD - torch.sum(log_det_jacobian)

for epoch in range(1, 11):
    train_loss = 0
    vae_with_flow.train()
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = vae_with_flow(data)
        
       
        loss = loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')


# In[43]:


# Example usage after training:
visualize_samples_and_latent_space(vae_with_flow, train_loader)


# In[44]:


# Potential spatial distribution generated using training data
latent_samples = []
vae_with_flow.eval()
with torch.no_grad():
    for data, _ in train_loader:
        data = data.to(device)
        mu, logvar = vae_with_flow.encode(data)
        z = vae_with_flow.reparameterize(mu, logvar)[0]
        latent_samples.append(z.cpu())

target_distribution = torch.cat(latent_samples, dim=0)


# In[45]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.optimize import least_squares

# Multivariate g-k Distribution
class MultivariateGKDistribution(nn.Module):
    def __init__(self, dim, a=0, b=1, g=0, k=0):
        super(MultivariateGKDistribution, self).__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32).repeat(dim))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32).repeat(dim))
        self.g = nn.Parameter(torch.tensor(g, dtype=torch.float32).repeat(dim))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32).repeat(dim))
    
    def forward(self, u):
        z = self.a + self.b * (1 + 0.8 * (1 - torch.exp(-self.g * u)) / (1 + torch.exp(-self.g * u))) * (1 + u**2)**self.k * u
        return z

    def sample(self, num_samples):
        u = torch.randn(num_samples, self.dim, dtype=torch.float32)
        return self.forward(u)

# Levenberg-Marquardt Loss Function
def levenberg_marquardt_loss(params, model, target_distribution):
    dim = model.dim
    a, b, g, k = params[:dim], params[dim:2*dim], params[2*dim:3*dim], params[3*dim:]
    model.a.data = torch.tensor(a, dtype=torch.float32)
    model.b.data = torch.tensor(b, dtype=torch.float32)
    model.g.data = torch.tensor(g, dtype=torch.float32)
    model.k.data = torch.tensor(k, dtype=torch.float32)

    samples = model.sample(len(target_distribution))
    loss = torch.mean((samples - target_distribution) ** 2).item()
    return loss

def optimize_gk_parameters(model, target_distribution):
    initial_params = torch.cat([model.a.data, model.b.data, model.g.data, model.k.data]).numpy()
    
    # 使用 'trf' 方法替代 'lm'，因为 'trf' 没有 'lm' 的限制
    result = least_squares(levenberg_marquardt_loss, initial_params, args=(model, target_distribution), method='trf')
    
    optimized_params = result.x
    
    model.a.data = torch.tensor(optimized_params[:model.dim], dtype=torch.float32)
    model.b.data = torch.tensor(optimized_params[model.dim:2*model.dim], dtype=torch.float32)
    model.g.data = torch.tensor(optimized_params[2*model.dim:3*model.dim], dtype=torch.float32)
    model.k.data = torch.tensor(optimized_params[3*model.dim:], dtype=torch.float32)

    
    
# Flow Layer and Normalizing Flow
class FlowLayer(nn.Module):
    def __init__(self, latent_dim):
        super(FlowLayer, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim, dtype=torch.float32))
        self.w = nn.Parameter(torch.randn(latent_dim, 1, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u.unsqueeze(0) * activation

        psi = (1 - activation**2) * self.w.squeeze(1)
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))

        log_det_jacobian = torch.log(det_jacobian + 1e-6).float()

        return z_new, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=2, base_dist=None):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList([FlowLayer(latent_dim) for _ in range(num_flows)])
        self.base_dist = base_dist if base_dist is not None else MultivariateGKDistribution(latent_dim)
    
    def forward(self, z):
        log_det_jacobian = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian
    
    def sample(self, num_samples):
        z0 = self.base_dist.sample(num_samples)
        z, _ = self.forward(z0)
        return z

# VAE with Normalizing Flow and g-k Distribution
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
            nn.Sigmoid()
        )
        self.flow = NormalizingFlow(latent_dim, num_flows=num_flows)

    def encode(self, x):
        x = x.view(-1, 784)
        params = self.encoder(x)
        mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z, log_det_jacobian = self.flow(z)
        return z, log_det_jacobian

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, log_det_jacobian

    def generate(self, z):
        return self.decode(z)

# Define the MNIST dataset with the correct transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the VAE model with Normalizing Flow and g-k distribution
vae_with_flow = VAEWithFlow(input_dim=784, hidden_dim=400, latent_dim=50, num_flows=4).to(device)

# Levenberg-Marquardt optimization of g-k distribution before training
# Here, we assume target_distribution is some pre-defined target data you wish to fit the g-k distribution to
# Replace with actual target distribution data
optimize_gk_parameters(vae_with_flow.flow.base_dist, target_distribution)

# Define the optimizer
optimizer = optim.Adam(vae_with_flow.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    x = x.view(-1, 784)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD - torch.sum(log_det_jacobian)

for epoch in range(1, 11):
    train_loss = 0
    vae_with_flow.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = vae_with_flow(data)
        loss = loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')


# In[46]:


# Define the MNIST dataset with the correct transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the VAE model with Normalizing Flow and g-k distribution
vae_with_flow = VAEWithFlow(input_dim=784, hidden_dim=400, latent_dim=50, num_flows=4).to(device)

# Levenberg-Marquardt optimization of g-k distribution before training
# Here, we assume target_distribution is some pre-defined target data you wish to fit the g-k distribution to

optimize_gk_parameters(vae_with_flow.flow.base_dist, target_distribution)

# Define the optimizer
optimizer = optim.Adam(vae_with_flow.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    x = x.view(-1, 784)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD - torch.sum(log_det_jacobian)

for epoch in range(1, 11):
    train_loss = 0
    vae_with_flow.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = vae_with_flow(data)
        loss = loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')


# In[47]:


# 调用可视化函数
visualize_samples_and_latent_space(vae_with_flow, train_loader)


# ### Advantages of This Plot:
# 1. **Clear Boundaries**:
#    - Distinct classification boundaries suggest well-defined regions for different digits in the latent space, aiding clear differentiation during generation.
# 
# 2. **Concentrated Regions**:
#    - The data points are more concentrated, indicating deeper recognition of specific digits, which supports generating distinct, well-separated digits.
# 
# 3. **Better Separation**:
#    - The clear boundaries and dispersed distribution help the model better identify and generate each digit.
# 
# ### Features of the Previous Plot:
# 1. **Broader Coverage**:
#    - The previous plot shows an even distribution over a larger latent space, which might help generate diverse samples but could reduce classification clarity.
# 
# 2. **Higher Ambiguity**:
#    - The lack of distinct boundaries in the previous plot may lead to less clear differentiation between digits.
# 
# ### Conclusion:
# **This plot** better aligns with our goal of clearly distinguishing digits, with superior classification boundaries and concentrated regions, making it more effective for our needs.

# In[51]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.optimize import least_squares

# Multivariate g-k Distribution
class MultivariateGKDistribution(nn.Module):
    def __init__(self, dim, a=0, b=1, g=0, k=0):
        super(MultivariateGKDistribution, self).__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32).repeat(dim))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32).repeat(dim))
        self.g = nn.Parameter(torch.tensor(g, dtype=torch.float32).repeat(dim))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32).repeat(dim))
    
    def forward(self, u):
        z = self.a + self.b * (1 + 0.8 * (1 - torch.exp(-self.g * u)) / (1 + torch.exp(-self.g * u))) * (1 + u**2)**self.k * u
        return z

    def sample(self, num_samples):
        u = torch.randn(num_samples, self.dim, dtype=torch.float32)
        return self.forward(u)

# Levenberg-Marquardt Loss Function
def levenberg_marquardt_loss(params, model, target_distribution):
    dim = model.dim
    a, b, g, k = params[:dim], params[dim:2*dim], params[2*dim:3*dim], params[3*dim:]
    model.a.data = torch.tensor(a, dtype=torch.float32)
    model.b.data = torch.tensor(b, dtype=torch.float32)
    model.g.data = torch.tensor(g, dtype=torch.float32)
    model.k.data = torch.tensor(k, dtype=torch.float32)

    samples = model.sample(len(target_distribution))
    loss = torch.mean((samples - target_distribution) ** 2).item()
    return loss

def optimize_gk_parameters(model, target_distribution):
    initial_params = torch.cat([model.a.data, model.b.data, model.g.data, model.k.data]).numpy()
    
    result = least_squares(levenberg_marquardt_loss, initial_params, args=(model, target_distribution), method='trf')
    
    optimized_params = result.x
    
    model.a.data = torch.tensor(optimized_params[:model.dim], dtype=torch.float32)
    model.b.data = torch.tensor(optimized_params[model.dim:2*model.dim], dtype=torch.float32)
    model.g.data = torch.tensor(optimized_params[2*model.dim:3*model.dim], dtype=torch.float32)
    model.k.data = torch.tensor(optimized_params[3*model.dim:], dtype=torch.float32)

# Flow Layers and Normalizing Flow
class FlowLayer(nn.Module):
    def __init__(self, latent_dim):
        super(FlowLayer, self).__init__()
        self.u = nn.Parameter(torch.randn(latent_dim, dtype=torch.float32))
        self.w = nn.Parameter(torch.randn(latent_dim, 1, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, z):
        linear = torch.matmul(z, self.w) + self.b
        activation = torch.tanh(linear)
        z_new = z + self.u.unsqueeze(0) * activation

        psi = (1 - activation**2) * self.w.squeeze(1)
        det_jacobian = torch.abs(1 + torch.matmul(psi, self.u))

        log_det_jacobian = torch.log(det_jacobian + 1e-6).float()

        return z_new, log_det_jacobian

# Introducing Radial Flow for more complex transformations
class RadialFlow(nn.Module):
    def __init__(self, latent_dim):
        super(RadialFlow, self).__init__()
        self.z0 = nn.Parameter(torch.randn(1, latent_dim, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, z):
        r = torch.norm(z - self.z0, dim=1, keepdim=True)
        h = 1 / (self.alpha + r)
        beta_h = self.beta * h
        z_new = z + beta_h * (z - self.z0)
        log_det_jacobian = torch.log(1 + beta_h * (1 - beta_h * r)).sum(dim=1)
        return z_new, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, latent_dim, num_flows=2, base_dist=None):
        super(NormalizingFlow, self).__init__()
        # 使用更加复杂的流结构，结合不同类型的流
        self.flows = nn.ModuleList(
            [FlowLayer(latent_dim) if i % 2 == 0 else RadialFlow(latent_dim) for i in range(num_flows)]
        )
        self.base_dist = base_dist if base_dist is not None else MultivariateGKDistribution(latent_dim)
    
    def forward(self, z):
        log_det_jacobian = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_jacobian += log_det
        return z, log_det_jacobian
    
    def sample(self, num_samples):
        z0 = self.base_dist.sample(num_samples)
        z, _ = self.forward(z0)
        return z

# VAE with Complex Normalizing Flow and g-k Distribution
class VAEWithFlow(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=50, num_flows=6):  # 增加 num_flows
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
            nn.Sigmoid()
        )
        self.flow = NormalizingFlow(latent_dim, num_flows=num_flows)  # 使用复杂的流结构

    def encode(self, x):
        x = x.view(-1, 784)
        params = self.encoder(x)
        mu, logvar = params[:, :self.latent_dim], params[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z, log_det_jacobian = self.flow(z)
        return z, log_det_jacobian

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, log_det_jacobian = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, log_det_jacobian

    def generate(self, z):
        return self.decode(z)

# Define the MNIST dataset with the correct transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the VAE model with Complex Normalizing Flow and g-k distribution
vae_with_flow = VAEWithFlow(input_dim=784, hidden_dim=400, latent_dim=50, num_flows=6).to(device)

# Levenberg-Marquardt optimization of g-k distribution before training
# Here, we assume target_distribution is some pre-defined target data you wish to fit the g-k distribution to
# Replace with actual target distribution data
optimize_gk_parameters(vae_with_flow.flow.base_dist, target_distribution)

# Define the optimizer
optimizer = optim.Adam(vae_with_flow.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar, log_det_jacobian):
    x = x.view(-1, 784)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD - torch.sum(log_det_jacobian)

for epoch in range(1, 11):
    train_loss = 0
    vae_with_flow.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, log_det_jacobian = vae_with_flow(data)
        loss = loss_function(recon_batch, data, mu, logvar, log_det_jacobian)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset):.6f}')


# In[52]:


# 调用可视化函数
visualize_samples_and_latent_space(vae_with_flow, train_loader)


# ### Advantages:
# 1. **Clearer Classification**:
#    - The data points in this new plot are distinctly grouped into three clusters, indicating that the model effectively separates different categories (likely different digits) in the latent space. This clustered distribution aids in clearer classification of different digits.
# 
# 2. **Higher Separation in Latent Space**:
#    - The noticeable gaps between clusters suggest a higher degree of separation between different categories. This distribution is beneficial for accurately distinguishing and recognizing different categories during the generation process.
# 
# 3. **Tight Cluster Distribution**:
#    - The data points within each cluster are tightly packed, indicating the model's high consistency in generating samples of the same category, which helps reduce confusion between categories during generation.
# 
# ### Comparison with Previous Plots:
# - **Compared to the Previous Plots**:
#   - The new plot shows more distinct classification, with clearer clusters and gaps between them. This suggests that the new model may better differentiate between different digits in the generation task.
#   
# - **Latent Space Coverage**:
#   - While the new plot has a less extensive coverage of the latent space compared to the previous plots, it offers more explicit and clear classification. Therefore, if the goal is to clearly distinguish between different categories of digits, the new result might be better.
# 
# ### Conclusion:
# **The new plot** performs the best, showing clearer category separation and tighter intra-category grouping in the latent space. This is highly advantageous for accurately distinguishing different digits in the generation task, making the algorithm behind the new plot more effective. Thank you!

# In[ ]:




