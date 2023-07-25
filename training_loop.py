import os
from datetime import datetime

import torch
from torch import nn, optim
from torchvision.utils import save_image
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from load_data import get_data_loader

# Instantiate G and D
nz = 100  # size of the latent z vector
num_classes = 26  # for 26 letters
G = Generator(nz, num_classes)
D = Discriminator(num_classes)

# Loss function
criterion = nn.BCELoss()

# Convention for real and fake labels
REAL = 1
FAKE = 0

# Optimizers (fancy implementation of stochastic gradient descent)
lr = 0.0002  # learning rate
betas = (0.5, 0.999)  # beta1 and beta2 for Adam optimizer
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas)

# Number of training epochs
num_epochs = 50
# Stochastic gradient descent batch size
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
    # Training loop
    for epoch in range(num_epochs):
        errG, errD = None, None
        pbar = tqdm(data_loader)
        for i, (images, labels) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch + 1}")
            D.zero_grad()
            # Format batch
            # real_images = images.to(device)  # TODO
            real_images = images.unsqueeze(1).to(device)
            b_size = real_images.size(0)
            labels = labels.squeeze().long().to(device)
            output = D(real_images, labels).view(-1)
            # Pass REAL as the target for real images
            errD_real = criterion(output, torch.full((b_size,), REAL, dtype=torch.float, device=device))
            errD_real.backward()
            D_x = output.mean().item()

            # Generate fake image batch with Generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_labels = torch.randint(0, num_classes, (b_size,), device=device).long()
            # label_embeddings_fake = torch.eye(num_classes)[fake_labels].to(device)
            # fake = G(noise, label_embeddings_fake) # TODO
            fake = G(noise, fake_labels)

            # Classify all fake batch with Discriminator
            output = D(fake.detach(), fake_labels).view(-1)
            # Pass FAKE as the target for fake images
            errD_fake = criterion(output, torch.full((b_size,), FAKE, dtype=torch.float, device=device))
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update Discriminator
            optimizerD.step()

            ############################
            # (2) Update Generator network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            output = D(fake, fake_labels).view(-1)

            # Calculate Generator's loss based on this output
            errG = criterion(output, torch.full((b_size,), REAL, dtype=torch.float, device=device))

            # Calculate gradients for Generator
            errG.backward()

            # Update Generator
            optimizerG.step()

        # Output training stats
        print(f"[{epoch + 1}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

        if epoch % 1 == 0:  # save images every 5 epochs (adjust as necessary)
            # Prepare directory
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            dir_path = os.path.join('./output', f'{now}_epoch_{epoch}')
            os.makedirs(dir_path, exist_ok=True)

            with torch.no_grad():
                for letter_index in range(num_classes):
                    for i in range(4):  # generate four of each letter
                        noise = torch.randn(1, nz, 1, 1, device=device)
                        labels = torch.full((1,), letter_index, device=device)
                        label_embeddings = torch.eye(num_classes)[labels].to(device)
                        fake = G(noise, label_embeddings)
                        save_image(fake.detach(), os.path.join(dir_path, f'{chr(97 + letter_index)}_{i + 1}.png'))
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(G.state_dict(), f'G_{now}.pth')