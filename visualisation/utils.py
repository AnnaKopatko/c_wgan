import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os

import imageio

def make_gif(graph_type, network, root_dir):
    directory = os.path.join(root_dir, network, graph_type)
    output_filename = os.path.join(root_dir, network,  'gfts/')
    with imageio.get_writer(output_filename, mode='I') as writer:
        for i in range(len(os.listdir(directory))):
            filename = os.path.join(root_dir, network, graph_type,  str(i) + '.png')
            image = imageio.imread(filename)
            writer.append_data(image)

def save_tensor_images(image_tensor, step, network, root_dir, num_images=25, show=False):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.title('Epoch ' + step)
    output_filename = os.path.join(root_dir, network, 'image', step + '.png')
    plt.savefig(output_filename)
    if show:
        plt.show()

def save_losses_plots(step, gen_losses, disc_losses, network, root_dir, show = False):
    plt.plot(range(len(disc_losses)), disc_losses, label = 'Critic' if network == 'wgan' else 'Discriminator')
    plt.plot(range(len(gen_losses)), gen_losses, label = 'Generator')
    plt.title('Epoch ' + step)
    plt.legend(loc='lower right')
    output_filename = os.path.join(root_dir, network, 'losses', step + '.png')
    plt.savefig(output_filename)
    if show:
        plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
