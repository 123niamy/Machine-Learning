# GAN training workflow on Iris dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set working directory to script location
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# 1. Load and preprocess Iris
iris = load_iris()
X = iris.data
scaler = MinMaxScaler(feature_range=(-1,1))
X_scaled = scaler.fit_transform(X)
data = torch.tensor(X_scaled, dtype=torch.float32)

# 2. Generator - Creates fake data from random noise
class Generator(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.BatchNorm1d(16),  # Stabilizes training
            nn.ReLU(),
            nn.Linear(16, out_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
    def forward(self, z): return self.net(z)

# 3. Discriminator - Classifies real vs fake data
class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.LeakyReLU(0.2),  # Better gradients than ReLU
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    def forward(self, x): return self.net(x)

# 4. Training setup
z_dim = 8
G = Generator(z_dim, 4)
D = Discriminator(4)
opt_g = torch.optim.Adam(G.parameters(), lr=0.001)
opt_d = torch.optim.Adam(D.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary cross-entropy loss

# Track losses for visualization
d_losses, g_losses = [], []

# 5. Training loop
for epoch in range(2000):
    # Sample real data
    real = data[torch.randint(0, len(data), (32,))]
    # Sample noise
    z = torch.randn(32, z_dim)
    fake = G(z)

    # Discriminator update: maximize log(D(real)) + log(1 - D(fake))
    D_real = D(real)
    D_fake = D(fake.detach())
    loss_d_real = criterion(D_real, torch.ones_like(D_real))
    loss_d_fake = criterion(D_fake, torch.zeros_like(D_fake))
    loss_d = (loss_d_real + loss_d_fake) / 2
    opt_d.zero_grad(); loss_d.backward(); opt_d.step()

    # Generator update: maximize log(D(G(z)))
    D_fake = D(fake)
    loss_g = criterion(D_fake, torch.ones_like(D_fake))
    opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    # Track losses
    d_losses.append(loss_d.item())
    g_losses.append(loss_g.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: D_loss={loss_d.item():.4f}, G_loss={loss_g.item():.4f}")

# 6. Generate samples and visualize
print("\n✅ Training Complete!")

# Generate fake samples
with torch.no_grad():
    z_test = torch.randn(10, z_dim)
    fake_samples = G(z_test).numpy()
    # Inverse transform to original scale
    fake_samples = scaler.inverse_transform(fake_samples)

# Get real samples for comparison
real_samples = iris.data[:10]

# Print comparison
print("\n📊 Real vs Fake Iris Samples (first 5):")
print("Feature names:", iris.feature_names)
print("\nReal samples:")
print(real_samples[:5])
print("\nGenerated (fake) samples:")
print(fake_samples[:5])

# 7. Plot loss curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
plt.plot(g_losses, label='Generator Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Plot real vs fake feature comparison
plt.subplot(1, 2, 2)
plt.scatter(real_samples[:, 0], real_samples[:, 1], label='Real', alpha=0.6, s=100)
plt.scatter(fake_samples[:, 0], fake_samples[:, 1], label='Fake', alpha=0.6, s=100, marker='x')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Real vs Generated Iris Samples')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gan_iris_results.png', dpi=150, bbox_inches='tight')
print("\n📈 Visualization saved as 'gan_iris_results.png'")
plt.close()
