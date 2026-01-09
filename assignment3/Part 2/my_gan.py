import argparse
import os

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, conv = False):
        super(Generator, self).__init__()

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.latent_dim = latent_dim
        self.img_size = self.img_rows * self.img_cols * self.channels

        if conv:
            self.build_conv()
        else:
            self.build_mlp()

    def build_mlp(self):
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, self.img_size),  # output 28*28*1=784
            nn.Tanh(),
            nn.Unflatten(1, (self.channels, self.img_rows, self.img_cols))# (1, 28, 28)
        )
        
        # show model
        print("Generator Architecture:")
        print(self.model)

    def build_conv(self):
        # DCGAN-style generator for MNIST (1, 28, 28)
        # z -> Linear -> (128, 7, 7) -> upsample x2 -> (128, 14, 14) -> upsample x2 -> (64, 28, 28) -> (1, 28, 28)
        self.init_size = self.img_rows // 4  # 7
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128, momentum=0.8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        print("\nGenerator (DCGAN) Architecture:")
        print(self.l1)
        print(self.conv_blocks)

    def forward(self, z):
        # Generate images from z
        if hasattr(self, "conv_blocks"):
            out = self.l1(z)
            out = out.view(out.size(0), 128, self.init_size, self.init_size)
            return self.conv_blocks(out)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim, conv = False):
        super(Discriminator, self).__init__()

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.latent_dim = latent_dim
        self.img_size = self.img_rows * self.img_cols * self.channels

        if conv:
            self.build_conv()
        else:
            self.build_mlp()

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)

    def build_mlp(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        
        # show model
        print("\nDiscriminator Architecture:")
        print(self.model)

    def build_conv(self):
        # DCGAN-style discriminator for MNIST (1, 28, 28) -> logits
        # Use strided conv to downsample: 28->14->7
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(self.channels, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(128 * (self.img_rows // 4) * (self.img_cols // 4), 1)),
        )
        print("\nDiscriminator (DCGAN) Architecture:")
        print(self.model)

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    history = {
        "d_loss": [],
        "g_loss": [],
        "d_acc": [],
    }

    adversarial_loss = nn.BCEWithLogitsLoss()
    
    print("\nStarting Training\n")
    for epoch in range(args.n_epochs):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        d_acc_sum = 0.0
        num_batches = 0

        if epoch == 0:
            sample_images(generator, 0)

        for i, (imgs, _) in enumerate(dataloader):

            real_imgs = imgs.to(args.device)
            batch_size = real_imgs.size(0)
            
            # label
            valid = torch.ones(batch_size, 1, device=args.device)
            fake = torch.zeros(batch_size, 1, device=args.device)

            # Train Generator
            # -------------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, generator.latent_dim, device=args.device)
            g_loss = adversarial_loss(discriminator(generator(z)), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            # ---------------
            optimizer_D.zero_grad()

            real_pred = discriminator(real_imgs)
            real_loss = adversarial_loss(real_pred, valid)

            # Generate fake images without tracking gradients for G
            with torch.no_grad():
                z = torch.randn(batch_size, generator.latent_dim, device=args.device)
                gen_imgs_for_d = generator(z)

            fake_pred = discriminator(gen_imgs_for_d)
            fake_loss = adversarial_loss(fake_pred, fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # accuracy of discriminator (logits): sigmoid(x) > 0.5  <=>  x > 0
            real_accuracy = torch.mean((real_pred > 0.0).float()).item()
            fake_accuracy = torch.mean((fake_pred < 0.0).float()).item()
            d_accuracy = 0.5 * (real_accuracy + fake_accuracy)

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            d_acc_sum += float(d_accuracy)
            num_batches += 1

        # epoch metrics (mean over batches)
        if num_batches > 0:
            epoch_d_loss = d_loss_sum / num_batches
            epoch_g_loss = g_loss_sum / num_batches
            epoch_d_acc = d_acc_sum / num_batches
        else:
            epoch_d_loss = float("nan")
            epoch_g_loss = float("nan")
            epoch_d_acc = float("nan")

        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)
        history["d_acc"].append(epoch_d_acc)

        if epoch % 10 == 9:
            print(
                f"[Epoch {epoch+1}/{args.n_epochs}] "
                f"[D loss: {epoch_d_loss:.4f}, acc: {epoch_d_acc*100:.2f}%] "
                f"[G loss: {epoch_g_loss:.4f}]"
            )
            
        # save img sample
        if epoch % args.save_interval == args.save_interval - 1:
            sample_images(generator, epoch+1)
        
    return history


def plot_training_curves(history, out_path):
    epochs = np.arange(1, len(history["d_loss"]) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(epochs, history["d_loss"], label="D loss")
    axs[0].set_title("Discriminator Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(epochs, history["g_loss"], label="G loss", color="tab:orange")
    axs[1].set_title("Generator Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")

    axs[2].plot(epochs, np.array(history["d_acc"]) * 100.0, label="D acc")
    axs[2].set_title("Discriminator Accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy (%)")
    axs[2].set_ylim(0, 100)

    for ax in axs:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
            
def sample_images(generator, epoch):

    generator.eval()
    with torch.no_grad():
        z = torch.randn(25, generator.latent_dim, device=generator.device)
        gen_imgs = generator(z)
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(5, 5, figsize=(10, 10))
        cnt = 0
        for i in range(5):
            for j in range(5):
                img = gen_imgs[cnt].cpu().numpy() # (1, 28, 28)
                img = np.transpose(img, (1, 2, 0)) # (28, 28, 1)
                img = img.squeeze()  # (1, 28, 28) -> (28, 28)
                img = np.clip(img, 0, 1)
                
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.suptitle(f"Epoch {epoch}", fontsize=16)
        fig.tight_layout()
        fig.savefig(f"images/{epoch}.png")
        plt.close(fig)
    
    generator.train()


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim, conv=args.conv)
    discriminator = Discriminator(args.latent_dim, conv=args.conv)

    generator = generator.to(args.device)
    discriminator = discriminator.to(args.device)
    # used by sample_images()
    generator.device = args.device

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Start training
    history = train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    plot_training_curves(history, out_path=os.path.join("images", "training_curves.png"))

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    path = "models"
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(generator.state_dict(), f"{path}/generator.pth")
    torch.save(discriminator.state_dict(), f"{path}/discriminator.pth")
    print(f"Models saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument("--conv",action="store_true",
                        help="use DCGAN-style conv architectures for G/D",)
    parser.add_argument('--save_interval', type=int, default=20,
                        help='save every SAVE_INTERVAL epochs')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="training device: cuda, mps, or cpu",
    )
    args = parser.parse_args()
    print("using device:", args.device)
    main()
