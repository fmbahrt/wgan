import sys

import torch

from perceptual_cost import MultiChannelCost, L2
from dataset import celeba
from model import Critic, Generator, WGANLP, WWGAN

gen = Generator()
cri = Critic()

#wgan = WGANLP(gen, cri, cuda=torch.cuda.is_available())

g_metric = MultiChannelCost()
g_metric.load_state_dict(torch.load("./checkpoint/perceptual_cost/model_checkpoint.pt"))

wwgan = WWGAN(gen, cri, ground_metric=g_metric, cuda=torch.cuda.is_available())

dataloader = celeba(sys.argv[1], batch_size=128)

wwgan.train(dataloader, epochs=10, check_interval=1)
