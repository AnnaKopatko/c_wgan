import torch
from torch import nn

def critic_input(one_hot_class, image_dim, image_batch):
    '''
    Function for calculating the input to the discriminator, returns a (num_examples, width, height, channels + one_hot_class[1]) tensor
    Parameters:
    one_hot_class: a one hot encoded tensor with [num_examples, num_classes] dim
    image_dim: (width, height, channels) of an image
    '''
    #we add two additional dimentions resulting in: [batch_size, num_classes, 1, 1]
    one_hot_image = one_hot_class[:, :, None, None]
    #here we repeat the last two dimentions height and width times, in order to create an (batch_size, height, width, len(one_hot_class)) dims
    one_hot_image = one_hot_image.repeat(1, 1, image_dim[1], image_dim[2])
    #now we can combine the vectors
    combined = torch.cat((image_batch.float(), one_hot_image.float()), 1)
    return combined


def gradient_penalty(real_images, fake_images, eps, critic, critic_input, one_hot_class, image_dim):
    '''
    Parameters:
        critic: the critic model
        real_images: a batch of real images
        fake_images: a batch of fake images
        eps: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed images
    '''
    mixed_images = eps * real_images + (1 - eps) * fake_images

    #important note: we also need to combine the gradient penalty input with the one hot vector
    mixed_input = critic_input(one_hot_class, image_dim, mixed_images)

    mixed_scores = critic(mixed_input)
    gradient = \
    torch.autograd.grad(inputs=mixed_images, outputs=mixed_scores, grad_outputs=torch.ones_like(mixed_scores),
                        create_graph=True, retain_graph=True)[0]

    # Flatten the gradients so each row captures one image
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty



#init the weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)