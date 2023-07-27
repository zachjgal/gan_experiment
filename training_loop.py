import os
from datetime import datetime

import torch
from torch import nn, optim
from torchvision.utils import save_image
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from load_data import get_data_loader
from constants import nz, num_classes


# Instantiate G and D
G = Generator(nz, num_classes)
D = Discriminator(num_classes)

# Loss function
criterion = nn.BCELoss()

# Convention for real and fake labels
REAL = 1
FAKE = 0

# Learning rates
lr_d = 0.00015
lr_g = 0.0002
# Optimizers (fancy implementation of stochastic gradient descent)
betas = (0.5, 0.999)  # beta1 and beta2 for Adam optimizer
optimizerD = optim.Adam(D.parameters(), lr=lr_d, betas=betas)
optimizerG = optim.Adam(G.parameters(), lr=lr_g, betas=betas)

# Number of training epochs. Note that the model
# is saved regularly, so in practice I'll just
# stop training when I decide to
num_epochs = 1000000000
# Mini-batch stochastic gradient descent batch size
batch_size = 64

# Save samples from generator every _ epochs
sample_generation_interval = 1

# get loader for dataset
data_loader = get_data_loader(batch_size=batch_size, shuffle=True)

# Define the device
_device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(_device)
print(f"running on {_device}")

if __name__ == '__main__':
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir_path = os.path.join('./output', f'results_{now}')
    os.makedirs(out_dir_path, exist_ok=False)

    # An epoch goes through the entire training set.
    # There are 2400 of each letter class, making 62400 total real training images.
    for epoch in range(num_epochs):
        errG, errD = None, None
        pbar = tqdm(data_loader)
        # Load data in batches of `batch_size`
        for i, (images, labels) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch + 1}")
            # Zero out the gradient for the discriminator
            D.zero_grad()
            # Format real image batch
            real_images = images.unsqueeze(1).to(device)
            b_size = real_images.size(0)
            labels = labels.squeeze().long().to(device)

            # Discriminate real images; calculate gradient for batch
            output = D(real_images, labels).view(-1)
            errD_real = criterion(output, torch.full((b_size,), REAL, dtype=torch.float, device=device))
            errD_real.backward()

            # Generate fake image batch with Generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), dtype=torch.long, device=device)
            fake = G(noise, fake_labels)

            # Discriminate the fake images; calculate gradient for batch
            output = D(fake.detach(), fake_labels).view(-1)
            errD_fake = criterion(output, torch.full((b_size,), FAKE, dtype=torch.float, device=device))
            errD_fake.backward()

            # Combine gradients for real + fake for total loss
            errD = errD_real + errD_fake
            # Update Discriminator
            optimizerD.step()

            ############################
            # Update Generator x times for each discriminator step
            ###########################
            for _ in range(1):
                # Zero out generator gradient
                G.zero_grad()
                # Generate fake image batch
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_labels = torch.randint(0, num_classes, (b_size,), dtype=torch.long, device=device)
                fake = G(noise, fake_labels)
                output = D(fake, fake_labels).view(-1)
                # Calculate Generator's loss based on this output
                errG = criterion(output, torch.full((b_size,), REAL, dtype=torch.float, device=device))
                # Calculate gradients for Generator
                errG.backward()
                # Update Generator
                optimizerG.step()

        # Output loss for epoch
        print(f"[{epoch + 1}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

        if epoch % sample_generation_interval == 0:
            # Create output directory
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            out_path = os.path.join(out_dir_path, f'{now}_epoch_{epoch + 1}')
            os.makedirs(out_path, exist_ok=False)

            with torch.no_grad():
                for letter_index in range(num_classes):
                    for i in range(6):  # generate samples of each letter
                        noise = torch.randn(1, nz, 1, 1, device=device)
                        labels = torch.full((1,), letter_index, dtype=torch.long, device=device)
                        fake = G(noise, labels)
                        save_image(fake.detach(), os.path.join(out_path, f'{chr(97 + letter_index)}_{i + 1}.png'))
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            torch.save(G.state_dict(), os.path.join(out_path, f'G_{now}.pth'))
            torch.save(D.state_dict(), os.path.join(out_path, f'D_{now}.pth'))