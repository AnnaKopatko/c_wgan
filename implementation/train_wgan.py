import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm

from models import Generator, Discriminator
from utils import weights_init, critic_input, gradient_penalty
from loss import crit_loss, gen_loss
from visualisation.utils import save_losses_plots, save_tensor_images, running_mean

#the hyperparameters

critic_repeats = 5

n_epochs = 100
z_dim = 64
batch_size = 128
lr = 0.0002
c_lambda = 10
#for the optimizer
beta_1 = 0.5
beta_2 = 0.999

device = 'cuda'
num_classes = 10
image_dim = (1, 28, 28)
root_dir ='visualisation/output/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = torchvision.datasets.FashionMNIST('.', train= True, transform = transform,  download= True)

dataloader = DataLoader(dataset,
    batch_size=batch_size,
    shuffle=True)

generator = Generator(input_dim = z_dim + num_classes).to(device)
critic = Discriminator(im_chan = image_dim[0] + num_classes).to(device)

gen_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas = (beta_1, beta_2))
critic_opt = torch.optim.Adam(critic.parameters(), lr = lr, betas = (beta_1, beta_2))

generator = generator.apply(weights_init)
critic = critic.apply(weights_init)

critic_losses = []
gen_losses = []
cur_step = 0
for epoch in range(n_epochs):
    for real_images, labels in tqdm(dataloader):
        cur_batch_size = len(real_images)
        real_images = real_images.to(device)
        one_hot_class = F.one_hot(labels.to(device), num_classes)
        mean_iteration_critic_loss = 0
        for _ in range(critic_repeats):
            critic_opt.zero_grad()

            noise = torch.rand(cur_batch_size, z_dim, device=device)
            gen_input_1 = torch.cat((noise.float(), one_hot_class.float()), dim=1)
            fake_images_1 = generator(gen_input_1)

            fake_critic_input_1 = critic_input(one_hot_class, image_dim, fake_images_1.detach())
            fake_score_1 = critic(fake_critic_input_1)

            real_critic_input = critic_input(one_hot_class, image_dim, real_images)
            real_score = critic(real_critic_input)

            epsilon = torch.rand(len(real_images), 1, 1, 1, device=device, requires_grad=True)
            grad_penalty = gradient_penalty(real_images, fake_images_1.detach(), epsilon, critic, critic_input,
                                            one_hot_class, image_dim)
            critic_loss = crit_loss(fake_score_1, real_score, grad_penalty, c_lambda)
            mean_iteration_critic_loss += critic_loss.item() / critic_repeats
            critic_loss.backward(retain_graph=True)
            critic_opt.step()

        critic_losses += [mean_iteration_critic_loss]
        gen_opt.zero_grad()

        noise = torch.rand(cur_batch_size, z_dim, device=device)
        gen_input_2 = torch.cat((noise.float(), one_hot_class.float()), dim=1)

        fake_images_2 = generator(gen_input_2)
        critic_input_2 = critic_input(one_hot_class, image_dim, fake_images_2)
        crit_fake_pred_2 = critic(critic_input_2)

        generator_loss = gen_loss(crit_fake_pred_2)
        generator_loss.backward()
        gen_losses += [generator_loss.item()]

        gen_opt.step()

    save_tensor_images(fake_images_2, str(epoch), 'wgan', root_dir)
    save_losses_plots(str(epoch), running_mean(gen_losses, 20), running_mean(critic_losses, 20), 'wgan', root_dir)
    cur_step += 1