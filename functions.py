import torch
import torch.nn as nn

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input / torch.norm(input, p=None, dim=1,keepdim=True)

class AdaIn(nn.Module):
    def __init__(self, n_channels, latent_dimension):
        super().__init__()

        self.normalisation = nn.InstanceNorm2d(n_channels)
        self.style_transform = nn.Linear(latent_dimension,latent_dimension*2)
        self.style_transform.bias.data[n_channels:] = 1     #the style std_dev starts at 1, rather than 0

    def forward(self, x, latent_style):
        out = self.normalisation(x)
        latent_mean, latent_std = self.style_transform(
            latent_style).chunk(2, dim=1)

        return out*latent_std + latent_mean
