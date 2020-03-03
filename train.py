import torch
import torch.nn as nn
import model


def train(discriminator, generator, args):


if __name__ == "__main__":
    args = dict(
        latent_size = 128,
        batch_size = 16,
        learning_rate = 5e-3,
        
    )

    generator = nn.DataParallel(model.Generator(latent_size)).cuda()
    discriminator = nn.DataParallel(model.Discriminator(latent_size)).cuda()


