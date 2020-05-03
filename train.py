import sys
import argparse

import torch

from perceptual_cost import MultiChannelCost, L2
from dataset import celeba
from model import Critic, Generator, WGANLP, WWGAN

# Parse Arguments
parser = argparse.ArgumentParser(description="Wasserstein GAN Bsc Thesis")

parser.add_argument("--backbone", required=True)
parser.add_argument("--datadir", required=True)

args = parser.parse_args()

backbone = args.backbone
datadir  = args.datadir

# Init Training
cuda = torch.cuda.is_available()

gen = Generator()
cri = Critic()

gan = None

# Chose backbone based on argument
if backbone.lower() == "wganlp":
    gan = WGANLP(gen, cri, backbone.lower(), cuda=cuda)
elif backbone.lower() == "l2_wgan":
    g_metric = L2()
    g_metric = g_metric.cuda() if cuda else g_metric
    gan = WWGAN(gen, cri, backbone.lower(), ground_metric=g_metric, cuda=cuda)
elif backbone.lower() == "watson_wgan":
    g_metric = MultiChannelCost()
    g_metric.load_state_dict(torch.load("./checkpoint/perceptual_cost/model_checkpoint.pt"))
    g_metric = g_metric.cuda() if cuda else g_metric
    gan = WWGAN(gen, cri, backbone.lower(), ground_metric=g_metric, cuda=cuda)

dataloader = celeba(datadir, batch_size=128)

gan.train(dataloader, epochs=10, check_interval=20)
