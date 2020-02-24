import torch

from dataset import celeba
from model import Critic, Generator, WGAN

gen = Generator()
cri = Critic()

wgan = WGAN(gen, cri, cuda=torch.cuda.is_available())

dataloader = celeba('/home/frederik/Documents/diku/bscthesis/data/celeba')

wgan.train(dataloader, epochs=10)
