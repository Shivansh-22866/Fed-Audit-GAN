"""
DCGAN Generator for Fed-AuditGAN
=================================
Generates synthetic data for fairness auditing using Deep Convolutional GAN architecture.
Implements conditional generation for creating targeted fairness probes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    DCGAN Generator network for creating high-quality synthetic samples.
    Conditioned on class labels for controlled generation.
    Uses convolutional layers for better image quality.
    """
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.channels = img_shape[0]
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Calculate initial size for deconvolution
        self.init_size = img_shape[1] // 4  # For 28x28 -> 7, for 32x32 -> 8
        
        # First linear layer to get to initial feature map size
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim * 2, 128 * self.init_size ** 2)
        )
        
        # Convolutional blocks for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic samples
        
        Args:
            noise: Random noise vector [batch_size, latent_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Generated images [batch_size, *img_shape]
        """
        # Embed labels and concatenate with noise
        label_input = self.label_emb(labels)
        gen_input = torch.cat([noise, label_input], dim=1)
        
        # Project and reshape
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Generate image through conv blocks
        img = self.conv_blocks(out)
        
        return img


class Discriminator(nn.Module):
    """
    DCGAN Discriminator network.
    Distinguishes real from generated samples using convolutional layers.
    """
    
    def __init__(self, num_classes: int = 10, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.channels = img_shape[0]
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns downsampling layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        # Label embedding as additional channel
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels + num_classes, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate the size after convolutions
        # After 4 conv layers with stride 2 and padding 1:
        # For 28x28: 28 -> 14 -> 7 -> 4 -> 2 (final: 2x2)
        # For 32x32: 32 -> 16 -> 8 -> 4 -> 2 (final: 2x2)
        # Formula: output_size = floor((input_size + 2*padding - kernel) / stride) + 1
        # With kernel=3, stride=2, padding=1: output = floor((input + 2 - 3) / 2) + 1 = floor((input - 1) / 2) + 1
        ds_size = img_shape[1]
        for _ in range(4):  # 4 conv layers
            ds_size = (ds_size + 2 * 1 - 3) // 2 + 1
        
        # Output layer
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Classify images as real or fake
        
        Args:
            img: Input images [batch_size, *img_shape]
            labels: Class labels [batch_size]
            
        Returns:
            Validity scores [batch_size, 1]
        """
        # Embed labels and expand to image dimensions
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(
            label_embedding.shape[0], 
            self.num_classes, 
            1, 
            1
        )
        label_embedding = label_embedding.expand(
            -1, -1, self.img_shape[1], self.img_shape[2]
        )
        
        # Concatenate image and label embedding
        d_in = torch.cat([img, label_embedding], dim=1)
        
        # Pass through conv blocks
        out = self.conv_blocks(d_in)
        out = out.view(out.shape[0], -1)
        
        # Get validity score
        validity = self.adv_layer(out)
        
        return validity


def train_generator(
    generator: Generator,
    discriminator: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 50,
    device: str = 'cuda',
    lr: float = 0.0002,
    b1: float = 0.5,
    b2: float = 0.999,
    sample_interval: int = 10
) -> Tuple[Generator, Discriminator]:
    """
    Train the DCGAN generator and discriminator
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        dataloader: Training data loader
        n_epochs: Number of training epochs
        device: Device to train on
        lr: Learning rate
        b1: Adam beta1
        b2: Adam beta2
        sample_interval: Interval for logging
        
    Returns:
        Trained generator and discriminator
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    
    logger.info(f"Training DCGAN for {n_epochs} epochs on {device}...")
    
    for epoch in range(n_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0
        
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # Move data to device
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise and labels
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            gen_labels = torch.randint(0, generator.num_classes, (batch_size,), device=device)
            
            # Generate images
            gen_imgs = generator(z, gen_labels)
            
            # Loss measures generator's ability to fool discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1
        
        # Log epoch statistics
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / n_batches
        
        if (epoch + 1) % sample_interval == 0:
            logger.info(
                f"[Epoch {epoch+1}/{n_epochs}] "
                f"[D loss: {avg_d_loss:.4f}] "
                f"[G loss: {avg_g_loss:.4f}]"
            )
    
    logger.info("DCGAN training completed!")
    return generator, discriminator


def generate_synthetic_samples(
    generator: Generator,
    num_samples: int,
    device: str = 'cuda',
    target_class: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic samples using trained generator
    
    Args:
        generator: Trained generator network
        num_samples: Number of samples to generate
        device: Device to generate on
        target_class: Specific class to generate (None for random)
        
    Returns:
        Tuple of (generated_images, labels)
    """
    generator.eval()
    
    with torch.no_grad():
        # Sample noise
        z = torch.randn(num_samples, generator.latent_dim, device=device)
        
        # Generate labels
        if target_class is not None:
            labels = torch.full((num_samples,), target_class, dtype=torch.long, device=device)
        else:
            labels = torch.randint(0, generator.num_classes, (num_samples,), device=device)
        
        # Generate images
        gen_imgs = generator(z, labels)
    
    return gen_imgs, labels


# Legacy classes for backward compatibility
class FairnessProbeGenerator(nn.Module):
    """
    Generator that creates counterfactual fairness probes.
    
    Architecture:
    - Encoder: Compresses input into latent representation
    - Bottleneck: Low-dimensional latent space
    - Decoder: Reconstructs to original dimensionality
    
    The generator is trained adversarially against the frozen global model
    to find inputs that maximize prediction differences (fairness vulnerabilities).
    
    Args:
        input_dim (int): Dimension of input data (e.g., 784 for MNIST 28x28)
        hidden_dims (list): List of hidden layer dimensions for encoder/decoder
        sensitive_attrs_indices (list): Indices of sensitive attributes to modify
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 sensitive_attrs_indices=None, dropout_rate=0.2):
        super(FairnessProbeGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.sensitive_attrs_indices = sensitive_attrs_indices or []
        
        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck (latent space)
        bottleneck_dim = hidden_dims[-1] // 2
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        decoder_layers = []
        in_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate)
            ])
            in_dim = h_dim
        
        # Final layer to reconstruct input
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming normalized inputs [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """
        Generate counterfactual by modifying input.
        
        Args:
            x (torch.Tensor): Input samples [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Modified samples x' [batch_size, input_dim]
        """
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        latent = self.bottleneck(encoded)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Only modify sensitive attributes if specified
        if self.sensitive_attrs_indices:
            x_prime = x.clone()
            x_prime[:, self.sensitive_attrs_indices] = reconstructed[:, self.sensitive_attrs_indices]
        else:
            # Modify all attributes
            x_prime = reconstructed
            
        return x_prime
    
    def generate_probe_pairs(self, x):
        """
        Generate (original, counterfactual) probe pairs.
        
        Args:
            x (torch.Tensor): Input samples
            
        Returns:
            tuple: (original samples, counterfactual samples)
        """
        with torch.no_grad():
            x_prime = self.forward(x)
        return x, x_prime


class ConvolutionalGenerator(nn.Module):
    """
    Convolutional Generator for image data.
    Better suited for spatial data like CIFAR-10/100.
    
    Args:
        input_shape (tuple): Shape of input images (C, H, W)
        latent_dim (int): Dimension of latent space
    """
    
    def __init__(self, input_shape=(3, 32, 32), latent_dim=128):
        super(ConvolutionalGenerator, self).__init__()
        
        self.input_shape = input_shape
        channels, height, width = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 4x4
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened dimension
        self.flat_dim = 256 * (height // 8) * (width // 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decode_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Generate counterfactual images.
        
        Args:
            x (torch.Tensor): Input images [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Modified images [batch_size, C, H, W]
        """
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)
        flattened = encoded.view(batch_size, -1)
        
        # Bottleneck
        latent = self.bottleneck(flattened)
        
        # Decode
        decoded_flat = self.decode_fc(latent)
        decoded_reshaped = decoded_flat.view(batch_size, 256, 
                                             self.input_shape[1] // 8, 
                                             self.input_shape[2] // 8)
        x_prime = self.decoder(decoded_reshaped)
        
        return x_prime
