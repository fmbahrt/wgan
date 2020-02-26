#!/bin/python3
# normal cpu stuff: allocate cpus, memory
#SBATCH --cpus-per-task=4
#SBATCH -o "logfile-%j.out"
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run langer than 2 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=3:00:00

import torch
import sys
sys.path.append('./')

from dataset import celeba
from model import Critic, Generator, WGAN

gen = Generator()
cri = Critic()

wgan = WGAN(gen, cri, cuda=torch.cuda.is_available())

dataloader = celeba(sys.argv[1], batch_size=128)

wgan.train(dataloader, epochs=100)
