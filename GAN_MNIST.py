"""
GAN Training on MNIST Dataset
==============================
Generator produces fake digit images from noise vectors.
Discriminator distinguishes real MNIST digits from generated ones.
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# 2. HYPER-PARAMETERS
# ─────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 128
LATENT_DIM  = 100          # noise-vector dimension
LR          = 2e-4
BETAS       = (0.5, 0.999) # Adam moments (standard for GANs)
NUM_EPOCHS  = 200
SAVE_EPOCHS = [1, 50, 100, 150, 200]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR     = "gan_outputs"

# ─────────────────────────────────────────────
# 4. NETWORK DEFINITIONS  (top-level so worker
#    processes can import them safely on Windows)
# ─────────────────────────────────────────────

class Generator(nn.Module):
    """
    Maps a latent noise vector z ∈ R^LATENT_DIM  →  28×28 grayscale image.
    Architecture: fully-connected MLP with BatchNorm + LeakyReLU hidden layers,
    Tanh output (matches the [-1,1] normalisation of real images).
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # z → 256
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 → 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 1024 → 784  (= 28×28)
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.net(z)
        return img.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Maps a 28×28 image → scalar probability of being REAL.
    Architecture: fully-connected MLP with Dropout + LeakyReLU,
    Sigmoid output.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 784 → 1024
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # 1024 → 512
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # 512 → 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # 256 → 1
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.view(-1, 28 * 28)
        return self.net(x)


# ─────────────────────────────────────────────
# MAIN — all execution code must live here on
# Windows to prevent multiprocessing spawn from
# re-running the entire script in worker procs.
# ─────────────────────────────────────────────
if __name__ == '__main__':

    # ── 2b. Reproducibility ─────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Running on: {DEVICE}")

    # ── 3. Dataset preparation ───────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # scale to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0,  # 0 = main-process loading; avoids Windows multiprocessing spawn error
    )
    print(f"Dataset: {len(train_dataset)} images  |  Batches per epoch: {len(train_loader)}")

    # ── 5. Initialisation & optimizers ──────────
    generator     = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    optimizer_G = optim.Adam(generator.parameters(),     lr=LR, betas=BETAS)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    criterion = nn.BCELoss()   # Binary Cross-Entropy

    print("\nGenerator:")
    print(generator)
    print("\nDiscriminator:")
    print(discriminator)

    # Fixed noise for consistent visualisation across epochs
    fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

    # ── 6. Training loop ────────────────────────
    g_losses, d_losses = [], []

    def save_generated_images(epoch: int, noise: torch.Tensor):
        """Generate images from fixed noise and save as a grid PNG."""
        generator.eval()
        with torch.no_grad():
            fake_imgs = generator(noise).cpu()
        generator.train()

        grid = make_grid(fake_imgs, nrow=8, normalize=True, value_range=(-1, 1))
        img_np = grid.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Generator output — Epoch {epoch}", fontsize=14)
        path = os.path.join(OUT_DIR, f"epoch_{epoch:03d}.png")
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"  → Saved sample grid: {path}")


    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            batch_n   = real_imgs.size(0)

            # ── Real / fake labels ──────────────────
            real_labels = torch.ones(batch_n, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_n, 1, device=DEVICE)

            # ══════════════════════════════════════════
            # TRAIN DISCRIMINATOR
            # ══════════════════════════════════════════
            optimizer_D.zero_grad()

            # (a) Real data forward pass
            real_outputs = discriminator(real_imgs)
            d_loss_real  = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # (b) Fake data forward pass
            noise      = torch.randn(batch_n, LATENT_DIM, device=DEVICE)
            fake_imgs  = generator(noise).detach()   # detach so G is not updated here
            fake_outputs = discriminator(fake_imgs)
            d_loss_fake  = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()

            # ══════════════════════════════════════════
            # TRAIN GENERATOR
            # ══════════════════════════════════════════
            optimizer_G.zero_grad()

            # Re-generate (don't reuse detached fakes)
            noise      = torch.randn(batch_n, LATENT_DIM, device=DEVICE)
            fake_imgs  = generator(noise)
            fake_outputs = discriminator(fake_imgs)

            # Generator wants D to output 1 ("real") for its fakes
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # Average over batches
        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  "
                  f"D Loss: {avg_d:.4f}  |  G Loss: {avg_g:.4f}")

        if epoch in SAVE_EPOCHS:
            save_generated_images(epoch, fixed_noise)

    # ── 7. Plot losses ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, NUM_EPOCHS + 1), g_losses, label="Generator Loss",     color="#e74c3c", lw=1.5)
    ax.plot(range(1, NUM_EPOCHS + 1), d_losses, label="Discriminator Loss", color="#2980b9", lw=1.5)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Average BCE Loss", fontsize=13)
    ax.set_title("GAN Training Losses over 200 Epochs", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    loss_path = os.path.join(OUT_DIR, "loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLoss curve saved: {loss_path}")

    # Save raw loss arrays for reference
    np.save(os.path.join(OUT_DIR, "g_losses.npy"), np.array(g_losses))
    np.save(os.path.join(OUT_DIR, "d_losses.npy"), np.array(d_losses))

    # ── 8. Save model checkpoints ────────────────
    torch.save(generator.state_dict(),     os.path.join(OUT_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(OUT_DIR, "discriminator_final.pth"))
    print("Model weights saved.")
    print("\nTraining complete!")
