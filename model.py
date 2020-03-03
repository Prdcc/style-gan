import torch
from torchvision import transforms
import torch.nn as nn
from functions import *


class StyleDecoder(nn.Module):
    def __init__(self, latent_dimension, n_hidden_layers):
        super().__init__()

        deep_layers = [Normalize()]

        for i in range(n_hidden_layers):
            deep_layers.append(nn.Linear(latent_dimension, latent_dimension))
            deep_layers.append(nn.LeakyReLU(0.2))

        self.mlp = nn.Sequential(*deep_layers)

    def forward(self, input):
        return self.mlp(input)


class Constant(nn.Module):
    def __init__(self, n_channels, resolution=7):
        super().__init__()
        self.input = nn.Parameter(torch.randn(
            1, n_channels, resolution, resolution))

    def forward(self, batch_size):
        return self.input.repeat(batch_size, 1, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.ada = AdaIn(n_channels, n_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        # a lot of the entries in MNIST-Fashion are identically zero, relu gives more latituted to achieve this
        self.relu = nn.ReLU()

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.ada(input, style)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.ada(input, style)
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, latent_dimension=128, mlp_layers=8):
        super().__init__()
        self.latent_decoder = StyleDecoder(latent_dimension, mlp_layers)
        self.initial = Constant(latent_dimension)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res7 = ConvBlock(latent_dimension)
        self.res14 = ConvBlock(latent_dimension)
        self.res28 = ConvBlock(latent_dimension)
        self.n_channels = latent_dimension

    def forward(self, style_seed, res_lvl=0, alpha=1, batch_size=128):
        style = self.latent_decoder(style_seed)
        out = self.initial(batch_size)
        out = self.res7(out, style)
        if res_lvl == 0:
            return out

        out = self.upsample(out)
        new_out = self.res14(out, style)

        if res_lvl == 1:
            out = (1-alpha)*(out) + alpha*new_out
            return out

        out = self.upsample(new_out)
        new_out = self.res28(out, style)

        out = (1-alpha)*(out) + alpha*new_out
        return out


class Discriminator(nn.Module):
    def __init__(self, n_channels=128, dropout_chance=0.2):
        super().__init__()
        activation = nn.LeakyReLU(0.2)
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            activation,
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            activation
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            activation,
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            activation
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            activation,
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            activation
        )

        self.downsample = nn.AvgPool2d(2, stride=2)

        self.layer_out = nn.Sequential(
            nn.Dropout(dropout_chance),
            nn.Linear(n_channels*7*7, 120),
            activation,
            nn.Linear(120, 1)
        )

    def forward(self, input, res_lvl=0):
        out = self.conv_in(input)
        for i in range(res_lvl, 0, -1):
            out = self.downsample(out)
            out = self.conv2(out)

        out = out.view(out.size(0), -1)
        out = self.layer_out(out)
        return out