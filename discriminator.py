"""Discriminator model"""

import torch
from torch import nn

class Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        self.main = nn.Sequential(
            # First conv layer
            nn.Conv2d(
                in_channels=1 + num_classes,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Second conv layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Third conv layer
            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        c = self.embed(labels)
        c = c.view(c.size(0), self.num_classes, 1, 1)
        # we have to reshape the label embeddings to be
        # the size of the input, so basically we have to
        # repeat it along the height and width so that
        # equivalent information is given to the discriminator
        # regarding classification
        c = c.expand(c.size(0), self.num_classes, 28, 28)
        x = torch.cat([input, c], 1)
        return self.main(x)