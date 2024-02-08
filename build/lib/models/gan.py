"""
Implementation of Base GAN models.
"""
import torch
from ..modules import basemodel
from ..modules import losses

class BaseGenerator(basemodel.BaseModel):
    r"""
    Base class for a generic unconditional generator model.

    Attributes:
        channels (int): Number of channels in the input signal.
        nz (int): Latent size of noise and generated signal.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, channels, nz, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.nz = nz
        self.loss_type = loss_type

    def generate_signals(self, num_signals, device=None):
        r"""
        Generates num_signals randomly.

        Args:
            num_signals (int): Number of signals to generate
            device (torch.device): Device to send signals to.

        Returns:
            Tensor: A batch of generated signals.
        """
        if device is None:
            device = self.device

        noise = torch.randn((num_signals, self.channels, self.nz), device=device)
        fake_signals = self.forward(noise)

        return fake_signals

    def compute_gan_loss(self, output):
        r"""
        Computes GAN loss for generator.

        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).

        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        # Compute loss and backprop
        if self.loss_type == "gan":
            errG = losses.minimax_loss_gen(output)

        elif self.loss_type == "ns":
            errG = losses.ns_loss_gen(output)

        elif self.loss_type == "hinge":
            errG = losses.hinge_loss_gen(output)

        elif self.loss_type == "wasserstein":
            errG = losses.wasserstein_loss_gen(output)

        else:
            raise ValueError("Invalid loss_type {} selected.".format(self.loss_type))

        return errG

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real signals of shape (N, C, H, W). Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake signals
        fake_signals = self.generate_signals(num_signals=batch_size, device=device)

        # Compute output logit of D thinking signal real
        output = netD(fake_signals)

        # Compute loss
        errG = self.compute_gan_loss(output=output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class BaseDiscriminator(basemodel.BaseModel):
    r"""
    Base class for a generic unconditional discriminator model.

    Attributes:
        channels (int): Number of channels in the input signal.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, channels, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.loss_type = loss_type

    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real signals.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake signals.

        Returns:
            errD (Tensor): A batch of GAN losses for the discriminator.
        """
        # Compute loss for D
        if self.loss_type == "gan" or self.loss_type == "ns":
            errD = losses.minimax_loss_dis(output_fake=output_fake, output_real=output_real)

        elif self.loss_type == "hinge":
            errD = losses.hinge_loss_dis(output_fake=output_fake, output_real=output_real)

        elif self.loss_type == "wasserstein":
            errD = losses.wasserstein_loss_dis(output_fake=output_fake, output_real=output_real)

        else:
            raise ValueError("Invalid loss_type selected.")

        return errD

    def compute_probs(self, output_real, output_fake):
        r"""
        Computes probabilities from real/fake signals logits.

        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real signals.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake signals.

        Returns:
            tuple: Average probabilities of real/fake signal considered as real for the batch.
        """
        D_x = torch.sigmoid(output_real).mean().item()
        D_Gz = torch.sigmoid(output_fake).mean().item()

        return D_x, D_Gz

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real signals.
            netG (nn.Module): Generator model for obtaining fake signals.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()
        real_signals, real_labels = real_batch
        batch_size = real_signals.shape[0]  # Match batch sizes for last iter

        # Produce logits for real signals
        output_real = self.forward(real_signals)

        # Produce fake signals
        fake_signals = netG.generate_signals(num_signals=batch_size, device=device).detach()

        # Produce logits for fake signals
        output_fake = self.forward(fake_signals)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real, output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real, output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD.item(), group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
