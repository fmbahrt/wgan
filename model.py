import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio

import numpy as np
import torchvision.utils as vutils

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
 
        self.linear = nn.Sequential(
            nn.Linear(256, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128*16*16),
            nn.LeakyReLU()
        )

        self.conv   = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),

            nn.UpsamplingBilinear2d(size=32),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            
            nn.UpsamplingBilinear2d(size=64),
            nn.BatchNorm2d(64),
    
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 16, 16)
        x = self.conv(x)
        return x

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            
            nn.Conv2d(512, 1024, 4, stride=2, padding=1), 
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1, 4, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class WGAN():

    def __init__(
        self,
        generator,
        critic,
        cuda=True
    ):

        self.cuda = cuda

        self.G = generator
        self.C = critic
    
        self.g_opt = torch.optim.Adam(self.G.parameters(), 
                                      lr=1e-4,
                                      betas=(0.5, 0.9))
        
        self.c_opt = torch.optim.Adam(self.C.parameters(), 
                                      lr=1e-4,
                                      betas=(0.5, 0.9))

        self.G = self.G.cuda() if self.cuda else self.G
        self.C = self.C.cuda() if self.cuda else self.C

        self.seeds = torch.randn((64, 256))
        self.seeds = self.seeds.cuda() if self.cuda else self.seeds

        self.g_iters = 0

    def train(self, dataloader, epochs=100, n_critic=5, check_interval=1000):
        for i in range(epochs):
            self._run_epoch(dataloader,
                            n_critic=n_critic,
                            check_interval=check_interval)
            
            # Checkpoints
            self._sample_to_disk()
            torch.save(self.G.state_dict(), "./celeba-gen.pt")
            torch.save(self.C.state_dict(), "./celeba-cri.pt")

    def _run_epoch(self, dataloader, n_critic=5, check_interval=1000):
        for i, (real_data, _) in enumerate(dataloader): 
            batch_size = real_data.size(0)

            # Sample real data
            real_data = real_data.cuda() if self.cuda else real_data
            
            # Sample fake datga
            z = torch.randn((batch_size, 256))
            z = z.cuda() if self.cuda else z
            gen_data = self.G(z)

            c_loss, lp, l_grad = self._critic_step(real_data, gen_data, batch_size)

            if i % n_critic == 0:
                g_loss = self._generator_step(gen_data, batch_size)
                self.g_iters += 1

            if self.g_iters % check_interval == 0:
                print("-- GENERTOR STEP: {} -------------------".format(self.g_iters)) 
                print(" CRITIC LOSS           : {}".format(-c_loss))
                print(" GENERATOR LOSS        : {}".format(g_loss))
                print(" LIPSCHITZ PENALTY     : {}".format(lp))
                print(" LARGEST GRADIENT NORM : {}".format(l_grad))
                
                sys.stdout.flush()

                self._sample_to_disk()
                torch.save(self.G.state_dict(), "./celeba-gen.pt")
                torch.save(self.C.state_dict(), "./celeba-cri.pt")

    def _critic_step(self, data, gen_data, batch_size):
        self.c_opt.zero_grad()

        # detach gen_data such that we do not backprop throug
        #   generator
        gen_data = gen_data.detach()

        # Lipscitz Penalty
        lp, l_grad = self._lipschitz_penalty(data, gen_data, batch_size)

        # Calculate Loss
        c_loss = self.C(gen_data).mean() - self.C(data).mean() + lp
        
        # Backprop
        c_loss.backward()
        self.c_opt.step()

        return c_loss, lp, l_grad

    def _generator_step(self, gen_data, batch_size):
        self.g_opt.zero_grad()

        cri_pred = self.C(gen_data)

        # Calculate loss
        g_loss = -cri_pred.mean()

        # Backprop
        g_loss.backward()
        self.g_opt.step()

        return g_loss

    def _lipschitz_penalty(self, real, fake, batch_size, lamb=10):
        inter = self._sample_tau(real, fake, batch_size)

        c_preds = self.C(inter)

        g_outs = torch.ones(c_preds.size())
        g_outs = g_outs.cuda() if self.cuda else g_outs

        grads = torch.autograd.grad(
            outputs      = c_preds,
            inputs       = inter,
            grad_outputs = g_outs,
            create_graph = True,
            retain_graph = True
        )[0]

        # Flatten
        grads = grads.view(batch_size, -1)
        
        # Calculate gradient norms
        g_norms = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)
        l_grad = torch.max(g_norms) 

        # Time for LP :-)
        zeros   = torch.zeros(g_norms.size())
        zeros   = zeros.cuda() if self.cuda else zeros
        
        g_norms = torch.max(zeros, g_norms-1) ** 2
        lp = lamb * g_norms.mean()
        return lp, l_grad

    def _sample_tau(self, real, fake, batch_size):
        # Sample from the joint distribution

        # generate epsilons
        eps = torch.rand(batch_size, 1, 1, 1)
        eps = eps.cuda() if self.cuda else eps

        # interpolate 
        inter = eps * real + (1 - eps) * fake
        inter = inter.cuda() if self.cuda else inter
        inter.requires_grad=True 
        return inter
    
    def _sample_to_disk(self):
        samples = self.G(self.seeds)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.detach().cpu()
        
        grid = vutils.make_grid(samples, padding=2, normalize=True)
        vutils.save_image(grid, "./gan_sample.png") 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    gen = Generator()
    cri = Critic()
    print(cri)
    x = gen.forward(torch.randn((1, 256)))
    y = cri.forward(x)

    print(y.shape)

    #plt.imshow(np.transpose(y[0].detach().numpy(), (1, 2, 0)))
    #plt.show()
