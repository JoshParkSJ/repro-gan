"""
Base class implementation of WGAN-GP.
"""
import torch
from torch import autograd

from torch_mimicry.nets.gan import gan


class WGANGPBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        sequence_length (int): Starting width for upsampling generator output to an signal.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self,
                 nz,
                 ngf,
                 sequence_length,
                 loss_type='wasserstein',
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         sequence_length=sequence_length,
                         loss_type=loss_type,
                         **kwargs)

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
            real_batch (Tensor): A batch of real signals of shape (N, C, L).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)

        # Compute output logit of D thinking signal real
        output = netD(fake_signals)

        # Compute loss
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class WGANGPBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.        
    """
    def __init__(self, ndf, loss_type='wasserstein', gp_scale=10.0, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.gp_scale = gp_scale

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
            real_batch (Tensor): A batch of real signals of shape (N, C, L).
            netG (nn.Module): Generator model for obtaining fake signals.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Produce real signals
        real_signals, _ = real_batch
        batch_size = real_signals.shape[0]  # Match batch sizes for last iter

        # Produce fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()

        # Produce logits for real and fake signals
        output_real = self.forward(real_signals)
        output_fake = self.forward(fake_signals)

        # Compute losses
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        errD_GP = self.compute_gradient_penalty_loss(real_signals=real_signals,
                                                     fake_signals=fake_signals,
                                                     gp_scale=self.gp_scale)

        # Backprop and update gradients
        errD_total = errD + errD_GP
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data

    def compute_gradient_penalty_loss(self,
                                      real_signals,
                                      fake_signals,
                                      gp_scale=10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_signals (Tensor): A batch of real signals of shape (N, 3, L). // TODO: make num of channels configurable
            fake_signals (Tensor): A batch of fake signals of shape (N, 3, L).
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
        alpha = alpha.view(N, 3, L)
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
