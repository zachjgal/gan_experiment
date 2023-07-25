import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, nz, num_classes):
        super(Generator, self).__init__()
        # nz is the size of the noise vector that the G takes as input. It's
        # selected arbitrarily, and could be seen as a hyperparameter to tune. Because
        # this is a CGAN, we will concatenate to this input a representation of the
        # class we want.
        self.nz = nz
        self.num_classes = num_classes
        # Create an embedding "layer" that will convert the label into a dense vector
        # of the same dimensionality (num_classes). It will allow us to concatenate
        # this label embedding with the noise vector.
        self.embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        # Define the main G model
        self.main = nn.Sequential(

            # ignoring batch size (4th dimension) we have initial input size
            # nz + num_classes, 1, 1) our input vector.
            #
            # Note: This is how to calculate the size of the output after
            # applying a reverse convolution
            # output_size = (input_size - 1) * stride -
            #               2 * padding + kernel_size + output_padding
            #             = (1-1)*1 - 2*0 + 6 + 0 = 6
            #               |||||||||||||||||||||||...||||||||||||||||||||||||||||
            #               |||||||||||||||||||||||...||||||||||||||||||||||||||||
            #  |||||||||| < |||||||||||||||||||||||...||||||||||||||||||||||||||||
            #               |||||||||||||||||||||||...||||||||||||||||||||||||||||
            #               |||||||||||||||||||||||...||||||||||||||||||||||||||||
            #               |||||||||||||||||||||||...||||||||||||||||||||||||||||
            nn.ConvTranspose2d(
                in_channels=nz + num_classes,
                out_channels=256,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False
            ),
            # Batch Normalization normalizes the output from the previous layer
            # to have zero mean and unit variance. This helps in stabilizing training.
            nn.BatchNorm2d(num_features=256),
            # ReLU activation function, which makes negative values 0.
            nn.ReLU(inplace=True),

            # Repeat the same pattern of ConvTranspose2d -> BatchNorm2d -> ReLU with
            # adjusted parameters.
            # output_size = (input_size - 1) * stride -
            #               2 * padding + kernel_size + output_padding
            #             = (6-1)*2 - 0 + 4
            #             = 14
            #
            #                                                                    x128
            #                                                          ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            #                      x256                                ||||||||||...||||||||||
            # ||||||||||||||||||||||...|||||||||||||||||||||||||||     ||||||||||...||||||||||
            # ||||||||||||||||||||||...|||||||||||||||||||||||||||  /  ||||||||||...||||||||||
            # ||||||||||||||||||||||...||||||||||||||||||||||||||| /   ||||||||||...||||||||||
            # ||||||||||||||||||||||...||||||||||||||||||||||||||| \   ||||||||||...||||||||||
            # ||||||||||||||||||||||...|||||||||||||||||||||||||||  \  ||||||||||...||||||||||
            # ||||||||||||||||||||||...|||||||||||||||||||||||||||     ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            #                                                          ||||||||||...||||||||||
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            # Output layer size 28x28 to match emnist letters data set
            # Note that if you wanted a color image, you could use one
            # output channel for each of R, G, B. We want a greyscale
            # image, so we only need one channel.
            # output_size = (input_size - 1) * stride -
            #               2 * padding + kernel_size + output_padding
            #             = (14-1)*2 - 2*1 + 4
            #             = 26 - 2 + 4
            #             = 28
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            # Tanh function squashes pixel values between -1 and 1.
            # This is standard practice for GAN's and makes training easier.
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Create an embedding label for each sample in the batch, and
        # concatenate to our noise input for the batch
        c = self.embed(labels)
        c = c.view(c.size(0), self.num_classes, 1, 1)
        z = torch.cat([noise, c], 1)
        return self.main(z)