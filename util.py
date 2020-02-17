import matplotlib.pyplot as plt
import numpy as np
import torch as nn
from torch.utils.data import TensorDataset


def read_dataset(resolution=28, train=True):
    return TensorDataset(nn.load("data/%d/%s.pt"%(resolution,"training" if train else "test")))

def get_image_from_index(dataset, index):
    img = dataset[index]
    return img[0].numpy()

def plot_image(img):
    """
    Plots a single image
    """
    ax = plt.gca()
    ax.imshow(img, cmap = "Greys", aspect="equal")
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)
    plt.show()
