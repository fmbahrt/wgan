import sys

import torch

from dataset import celeba
from model import Critic, Generator, WGAN

gen = Generator()
cri = Critic()

wgan = WGAN(gen, cri, cuda=torch.cuda.is_available())

dataloader = celeba(sys.argv[1], batch_size=256)

wgan.train(dataloader, epochs=10)
