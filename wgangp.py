import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.fft import rfft, irfft
from torch.nn.functional import normalize
from scipy.signal import welch
from scipy import signal
import math
from torch.autograd import Variable


class Generator(WGANGPBaseGenerator):
    r"""
    Base class for a generic unconditional generator model.
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, **kwargs):
        super().__init__(nz=500, ngf=150, sequence_length=54)
        self.errG_array = []
        self.count = 0

        # Build the layers
        self.l1 = nn.Linear(self.nz, self.sequence_length * self.ngf)
        self.block1 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block5 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block6 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block7 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block8 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block9 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block10 = GBlock(self.ngf, self.ngf, upsample=False)
        self.block11 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block12 = GBlock(self.ngf, self.ngf, upsample=False)
        self.c13 = nn.Conv1d(self.ngf, 64, 1, 1, 0)
        self.end = nn.Linear(3204, 3152)

        # Initialise the weights
        nn.init.normal_(self.l1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c13.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, L).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], self.ngf, self.sequence_length)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.block8(h)
        h = self.block9(h)
        h = self.block10(h)
        h = self.block11(h)
        h = self.block12(h)
        h = self.c13(h)
        h = self.end(h)
        return h

    def generate_signals(self, num_signals, device=None):
        r"""
        Generates num_signals randomly.

        Args:
            num_signals (int): Number of signals to generate
            device (torch.device): Device to send images to.

        Returns:
            Tensor: A batch of generated images.
        """
        if device is None:
            device = self.device

        noise = torch.randn((num_signals, self.nz), device=device)
        fake_signals = self.forward(noise)

        return fake_signals

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, L).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        batch_size = real_batch[0].shape[0]  # Get only batch size from real batch

        # Produce logits for fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)
        out_fake = netD(fake_signals)

        self.zero_grad()

        # Backprop and update gradients
        errG = self.compute_gan_loss(out_fake)
        errG.backward()
        optG.step()

        # Log statistics
        self.errG_array.append(errG.item())
        if (self.count != 0 and self.count % 100 == 0):
            print(errG.item(), 'gen')
        self.count += 1
        log_data.add_metric('errG', errG.item(), group='loss')
        return log_data

    
def get_fft_feature_train(data, nperseg=256, noverlap=128, channels=64):
    all_fft = []
    device = data.device

    # Create window function
    window = torch.hann_window(nperseg, dtype=torch.float, device=device)

    for x in data:
        avg_psds_db = []
        
        for ch in range(channels):
            chx = x[ch]

            # Separate x into overlapping segments
            x_segs = chx.unfold(0, nperseg, nperseg - noverlap)

            # Apply window function to each segment
            windowed_segs = x_segs * window

            # Compute power spectral density for each windowed segment
            seg_psds = torch.fft.rfft(windowed_segs, dim=1)
            seg_psds = torch.abs(seg_psds)**2

            # Average PSDs over all segments
            avg_psds = torch.mean(seg_psds, axis=0)

            # Convert to decibels
            avg_psds_db.append(torch.log10(avg_psds + 1e-10))

        avg_psds_db = torch.stack(avg_psds_db)
        all_fft.append(avg_psds_db)

    all_fft = torch.stack(all_fft, dim=0).to(device)
    return all_fft




class Discriminator(WGANGPBaseDiscriminator):
    def __init__(self, **kwargs):
        super().__init__(ndf=150)
        self.count = 0
        self.errD_array = []

        self.sblock1 = DBlock(64, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock2 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock3 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock4 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock5 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock6 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock7 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock8 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock9 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock10 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.sblock11 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=False, reflectionPad=True)
        self.sblock12 = DBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, downsample=True, reflectionPad=True)
        self.c = nn.Conv1d(self.ndf, 64, 1, 1, 0)
        self.end = nn.Linear(45, 1)
        
        nn.init.normal_(self.c.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = x.float()
        h = self.sblock1(x)
        h = self.sblock2(h)
        h = self.sblock3(h)
        h = self.sblock4(h)
        h = self.sblock5(h)
        h = self.sblock6(h)
        h = self.sblock7(h)
        h = self.sblock8(h)
        h = self.sblock9(h)
        h = self.sblock10(h)
        h = self.sblock11(h)
        h = self.sblock12(h)
        h = self.c(h)
        h = self.end(h)
        return h.view(h.shape[0], 64)

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (dict): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        batch_size = real_batch.shape[0] # Match batch sizes for last iter

        # Produce logits for real signals
        real_signals = real_batch
        out_real = self.forward(real_signals)

        # Produce logits for fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()
        out_fake = self.forward(fake_signals)

        # Reset the gradients to zero
        optD.zero_grad()

        # Backprop and update gradients
        errD = self.compute_gan_loss(output_real=out_real, output_fake=out_fake)
        errD_GP = self.compute_gradient_penalty_loss(real_signals=real_signals, fake_signals=fake_signals)

        errD_total = errD + errD_GP
        errD_total.backward()
        optD.step()


        # Log statistics
        if (self.count != 0 and self.count % 5 == 0):
            self.errD_array.append(errD_total.item())
            log_data.add_metric('errD', errD.item(), group='loss')
            log_data.add_metric('errD_GP', errD_GP.item(), group='loss')
        if (self.count != 0 and self.count % 500 == 0):
            print(errD_total.item(), 'disc')
        if (self.count % 5000 == 0):
            plot_everything(fake_signals, netG.errG_array, self.errD_array)
        self.count += 1

        return log_data

    def compute_gradient_penalty_loss(self,
                                      real_signals,
                                      fake_signals,
                                      gp_scale=10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py

        Args:
            real_signals (Tensor): A batch of real signals of shape (N, 1, L). // TODO: make num of channels configurable
            fake_signals (Tensor): A batch of fake signals of shape (N, 1, L).
            gp_scale (float): Gradient penalty lamda parameter.

        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, L = real_signals.shape
        device = real_signals.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_signals.nelement() / N)).contiguous()
        alpha = alpha.view(N, 64, L)  # TODO: MAKE CHANNEL VARIABLE
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake signals.
        interpolates = alpha * real_signals.detach() \
            + ((1 - alpha) * fake_signals.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty


    
    
    
# Data handling objects
if torch.cuda.is_available():
  print("hello GPU")
else:
  print("sadge")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dataloader = DataLoader(
    TensorDataset(data),
    batch_size=32,
    shuffle=True
)

def weights_init(model):
    for m in model.modules():
      if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


netD = Discriminator()
netG = Generator()

weights_init(netD)
weights_init(netG)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    netD = nn.DataParallel(netD)
    netG = nn.DataParallel(netG)

netD.to("cuda" if torch.cuda.is_available() else "cpu")
netG.to("cuda" if torch.cuda.is_available() else "cpu")

optD = optim.Adam(netD.parameters(), 0.0001, (0.5, 0.99))
optG = optim.Adam(netG.parameters(), 0.0001, (0.5, 0.99))


# Start training
trainer = Trainer(
    netD=netD.module,
    netG=netG.module,
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=1000000,
    lr_decay='linear',
    dataloader=dataloader,
    save_steps=5000,
    print_steps=100,
    log_dir='/kaggle/working/model',
    device=device)
trainer.train()


# epoch = global step / 2


