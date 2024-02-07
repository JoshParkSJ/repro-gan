import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from models import WGANGPBaseGenerator, WGANGPBaseDiscriminator
from training import Trainer
from modules import DBlock, GBlock

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
data = torch.tensor(np.load("./64ch_signals.npy")).detach()
dataloader = DataLoader(
    TensorDataset(data),
    batch_size=32,
    shuffle=True
)

class Discriminator(WGANGPBaseDiscriminator):
    def __init__(self, **kwargs):
        super().__init__(channels=64)

        self.sblock1 = DBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, downsample=True)
        self.sblock2 = DBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, downsample=True)
        self.sblock3 = DBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, downsample=True)
        self.conv = nn.Conv1d(self.ndf, 64, kernel_size=1, stride=1, padding=0)
        self.end = nn.Linear(3152, 1)
        
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = x.float()
        h = self.sblock1(x)
        h = self.sblock2(h)
        h = self.sblock3(h)
        h = self.c(h)
        h = self.end(h)
        return h.view(h.shape[0], 64)

class Generator(WGANGPBaseGenerator):
    def __init__(self, **kwargs):
        super().__init__(channels=64, nz=3152)

        # Build the layers
        self.block1 = GBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, upsample=False)
        self.block2 = GBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, upsample=False)
        self.block3 = GBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, upsample=False)
        self.conv = nn.Conv1d(self.ngf, 64, kernel_size=1, stride=1, padding=0)

        # Initialise the weights
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.conv(h)
        h = self.end(h)
        return h

netD = Discriminator().to("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to("cuda" if torch.cuda.is_available() else "cpu")

optD = optim.Adam(netD.parameters(), 0.0001, (0.5, 0.99))
optG = optim.Adam(netG.parameters(), 0.0001, (0.5, 0.99))

trainer = Trainer(
    netD=netD.module, # use .module to use GPU
    netG=netG.module, # remove .module to use CPU
    optD=optD,
    optG=optG,
    n_dis=5,
    num_steps=1000,
    dataloader=dataloader,
    save_steps=5000,
    print_steps=100,
    log_dir='./saved_states',
    device=device)
trainer.train()