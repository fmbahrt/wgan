import sys

import torch

from dataset import celeba
from model import Critic, Generator, WGAN

gen = Generator()
cri = Critic()

wgan = WGAN(gen, cri, cuda=torch.cuda.is_available())

dataloader = celeba(sys.argv[1], batch_size=128)

wgan.train(dataloader, epochs=10, check_interval=1)
