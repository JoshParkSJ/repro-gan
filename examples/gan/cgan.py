"""
Implementation of Base GAN models for a generic conditional GAN.
"""
import torch

from torch_mimicry.nets.gan import gan


class BaseConditionalGenerator(gan.BaseGenerator):
    r"""
    Base class for a generic conditional generator model.

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        sequence_length (int): Starting width for upsampling generator output to an signal.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, num_classes, nz, ngf, sequence_length, loss_type, **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         sequence_length=sequence_length,
                         loss_type=loss_type,
                         **kwargs)
        self.num_classes = num_classes

    def generate_signals(self, num_signals, c=None, device=None):
        r"""
        Generate signals with possibility for conditioning on a fixed class.

        Args:
            num_signals (int): The number of signals to generate.
            c (int): The class of signals to generate. If None, generates random signals.
            device (int): The device to send the generated signals to.

        Returns:
            tuple: Batch of generated signals and their corresponding labels.
        """
        if device is None:
            device = self.device

        if c is not None and c >= self.num_classes:
            raise ValueError("Input class to generate must be in the range [0, {})".format(self.num_classes))

        if c is None:
            fake_class_labels = torch.randint(low=0,
                                              high=self.num_classes,
                                              size=(num_signals, ),
                                              device=device)

        else:
            fake_class_labels = torch.randint(low=c,
                                              high=c + 1,
                                              size=(num_signals, ),
                                              device=device)

        noise = torch.randn((num_signals, self.nz), device=device)
        fake_signals = self.forward(noise, fake_class_labels)

        return fake_signals

    def generate_signals_with_labels(self, num_signals, c=None, device=None):
        r"""
        Generate signals with possibility for conditioning on a fixed class.
        Additionally returns labels.

        Args:
            num_signals (int): The number of signals to generate.
            c (int): The class of signals to generate. If None, generates random signals.
            device (int): The device to send the generated signals to.

        Returns:
            tuple: Batch of generated signals and their corresponding labels.
        """
        if device is None:
            device = self.device

        if c is not None and c >= self.num_classes:
            raise ValueError("Input class to generate must be in the range [0, {})".format(self.num_classes))

        if c is None:
            fake_class_labels = torch.randint(low=0,
                                              high=self.num_classes,
                                              size=(num_signals, ),
                                              device=device)

        else:
            fake_class_labels = torch.randint(low=c,
                                              high=c + 1,
                                              size=(num_signals, ),
                                              device=device)

        noise = torch.randn((num_signals, self.nz), device=device)
        fake_signals = self.forward(noise, fake_class_labels)

        return fake_signals, fake_class_labels

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
            real_batch (Tensor): A batch of real signals of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake signals and labels
        fake_signals, fake_class_labels = self.generate_signals_with_labels(
            num_signals=batch_size, device=device)

        # Compute output logit of D thinking signal real
        output = netD(fake_signals, fake_class_labels)

        # Compute loss and backprop
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')

        return log_data


class BaseConditionalDiscriminator(gan.BaseDiscriminator):
    r"""
    Base class for a generic conditional discriminator model.

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, num_classes, ndf, loss_type, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.num_classes = num_classes

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
            real_batch (Tensor): A batch of real signals of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake signals.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()

        real_signals, real_class_labels = real_batch
        batch_size = real_signals.shape[0]  # Match batch sizes for last iter

        # Produce logits for real signals
        output_real = self.forward(real_signals, real_class_labels)

        # Produce fake signals and labels
        fake_signals, fake_class_labels = netG.generate_signals_with_labels(
            num_signals=batch_size, device=device)
        fake_signals, fake_class_labels = fake_signals.detach(
        ), fake_class_labels.detach()

        # Produce logits for fake signals
        output_fake = self.forward(fake_signals, fake_class_labels)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
