
# Project Title
**Enhancing Variational Autoencoder with Normalizing Flows: A Multivariate g-k Distribution and Levenberg-Marquardt Optimization Approach for MNIST Data Generation and Reconstruction**

## Overview
This project aims to elevate the performance of a Variational Autoencoder (VAE) by integrating Normalizing Flows with a Multivariate g-k Distribution, optimized using the Levenberg-Marquardt (LM) algorithm. The enhancements are applied to the MNIST dataset to improve the quality and diversity of generated images.

### Key Components

1. **Multivariate g-k Distribution**:
   - The g-k distribution is used as the base distribution within the Normalizing Flows. This distribution is particularly powerful in modeling complex data structures, offering more flexibility than the standard normal distribution.

2. **Levenberg-Marquardt Optimization**:
   - The parameters of the g-k distribution are fine-tuned using the Levenberg-Marquardt algorithm, a robust technique for non-linear parameter estimation. This optimization enhances the flow’s ability to capture intricate patterns in the data, leading to superior performance in generation and reconstruction tasks.

### Implementation Details

- **VAE Architecture**:
  - **Encoder**: Compresses input images from the MNIST dataset into a latent space.
  - **Normalizing Flows**: Applies invertible flows to transform the latent space, leveraging the g-k distribution as the foundational structure.
  - **Decoder**: Reconstructs images from the transformed latent space, aiming for high fidelity and diversity in the output.

- **Training Process**:
  - The model is trained on the MNIST dataset, utilizing the enhanced latent space provided by Normalizing Flows and the g-k distribution. The result is a VAE capable of generating more realistic and complex image data.

## Usage
- **Training**: Adjust the hyperparameters in the training script as needed. The script will train the VAE with Normalizing Flows on the MNIST dataset.
- **Evaluation**: After training, evaluate the reconstructed and generated images to observe the improvements in model performance due to the g-k distribution and LM optimization.

## Results
- The enhanced VAE model shows significant improvements in generating realistic and diverse images, as evidenced by the visualizations and quantitative metrics included in the results folder.

## Contributions
- **Multivariate g-k Distribution Integration**: Introduced a flexible and powerful base distribution in the VAE’s latent space transformation.
- **Levenberg-Marquardt Optimization**: Applied a robust optimization technique to refine the distribution parameters, resulting in better model performance.


