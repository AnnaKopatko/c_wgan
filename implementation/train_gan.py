import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm

from models import Generator, Discriminator
from utils import weights_init, critic_input
from loss import criterion
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
disc = Discriminator(im_chan = image_dim[0] + num_classes).to(device)

gen_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas = (beta_1, beta_2))
disc_opt = torch.optim.Adam(disc.parameters(), lr = lr, betas = (beta_1, beta_2))

generator = generator.apply(weights_init)
disc = disc.apply(weights_init)

cur_step = 0
disc_losses = []
gen_losses = []
for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        one_hot_labels = F.one_hot(labels.to(device), num_classes).to(device)
        # (features, n_classes)
        disc_opt.zero_grad()
        # Get noise corresponding to the current batch_size
        noise = torch.rand(cur_batch_size, z_dim, device=device)
        gen_input = torch.cat((noise.float(), one_hot_labels.float()), dim=1)
        fake = generator(gen_input)

        fake = generator(gen_input).detach()
        fake_image_and_labels = critic_input(one_hot_labels, image_dim, fake)
        real_image_and_labels = critic_input(one_hot_labels, image_dim, real)
        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_losses += [disc_loss.item()]
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        ### Update generator ###
        # Zero out the generator gradients
        gen_opt.zero_grad()
        fake = generator(gen_input)
        fake_image_and_labels = critic_input(one_hot_labels, image_dim, fake)
        # This will error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_losses += [gen_loss.item()]
        gen_loss.backward()
        gen_opt.step()

    save_tensor_images(fake, str(epoch), 'gan', root_dir)
    save_losses_plots(str(epoch), running_mean(gen_losses, 20), running_mean(disc_losses, 20), 'gan', root_dir)
    cur_step += 1