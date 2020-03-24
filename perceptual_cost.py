import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Visualizer

EPS = 1e-10
#EPS = 1e-6

class Rfft2d(nn.Module):
    # This is not my code, credit: ?
    """
    Blockwhise 2D FFT
    for fixed blocksize of 8x8
    """
    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        """
        super(Rfft2d, self).__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        self.unfold = nn.Unfold(kernel_size=self.blocksize, padding=0, stride=self.stride)
        
    def forward(self, x):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, b, b/2, 2)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block real FFT coefficients. 
        The last dimension is pytorches representation of complex values
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "FFT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, 64, k)
        (N, _, k) = x.shape
        x = x.view(-1,self.blocksize,self.blocksize,k).permute(0,3,1,2)
        # now shape (N, #k, b, b)
        # perform DCT
        coeff = torch.rfft(x, signal_ndim=2)
        
        return coeff / self.blocksize**2

def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:,:,:,:,0] + b * softmax_factors[:,:,:,:,1]

class PerceptualCost2(nn.Module):
    """
    Loss function based on Watsons perceptual distance.
    Based on FFT quantization
    """
    def __init__(self, blocksize=8, trainable=True, reduction='none'):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()
        self.trainable = trainable
        
        # input mapping
        blocksize = torch.as_tensor(blocksize)
        
        # module to perform 2D blockwise rFFT
        self.add_module('fft', Rfft2d(blocksize=blocksize.item(), interleaving=False))
    
        # parameters
        self.weight_size = (blocksize, blocksize // 2 + 1)
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        # init with uniform QM
        self.t_tild = nn.Parameter(torch.zeros(self.weight_size), requires_grad=trainable)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=trainable) # luminance masking
        w = torch.tensor(0.2) # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1- w)), requires_grad=trainable) # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable) # pooling
        
        # phase weights
        self.w_phase_tild = nn.Parameter(torch.zeros(self.weight_size) -2., requires_grad=trainable)
        
        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)
        
        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception('Reduction "{}" not supported. Valid values are: "sum", "none".'.format(reduction))

    @property
    def t(self):
        # returns QM
        qm = torch.exp(self.t_tild)
        return qm
    
    @property
    def w(self):
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)
    
    @property
    def w_phase(self):
        # return weights for phase
        w_phase =  torch.exp(self.w_phase_tild)
        # set weights of non-phases to 0
        if not self.trainable:
            w_phase[0,0] = 0.
            w_phase[0,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1,self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, 0] = 0.
        return w_phase
    
    def forward(self, input, target):
        # fft
        c0 = self.fft(target)
        c1 = self.fft(input)
        
        N, K, H, W, _ = c0.shape
        
        # get amplitudes
        c0_amp = torch.norm(c0 + EPS, p='fro', dim=4)
        c1_amp = torch.norm(c1 + EPS, p='fro', dim=4)
        
        # luminance masking
        avg_lum = torch.mean(c0_amp[:,:,0,0])
        t_l = self.t.view(1, 1, H, W).expand(N, K, H, W)
        t_l = t_l * (((c0_amp[:,:,0,0] + EPS) / (avg_lum + EPS)) ** self.alpha).view(N, K, 1, 1)
        
        # contrast masking
        s = softmax(t_l, (c0_amp.abs() + EPS)**self.w * t_l**(1 - self.w))
        
        # pooling
        watson_dist = (((c0_amp - c1_amp) / s).abs() + EPS) ** self.beta
        watson_dist = self.dropout(watson_dist) + EPS
        watson_dist = torch.sum(watson_dist, dim=(1,2,3))
        watson_dist = watson_dist ** (1 / self.beta)
        
        # get phases
        c0_phase = torch.atan2( c0[:,:,:,:,1], c0[:,:,:,:,0] + EPS) 
        c1_phase = torch.atan2( c1[:,:,:,:,1], c1[:,:,:,:,0] + EPS)
        
        # angular distance
        phase_dist = torch.acos(torch.cos(c0_phase - c1_phase)*(1 - EPS*10**3)) * self.w_phase # we multiply with a factor ->1 to prevent taking the gradient of acos(-1) or acos(1). The gradient in this case would be -/+ inf
        phase_dist = self.dropout(phase_dist)
        phase_dist = torch.sum(phase_dist, dim=(1,2,3))
        
        # perceptual distance
        distance = watson_dist + phase_dist
        
        # reduce
        if self.reduction == 'sum':
            distance = torch.sum(distance)
        
        return distance
    
class PerceptualCost(nn.Module):

    def __init__(self, blocksize=8):
        super(PerceptualCost, self).__init__()

        self.add_module('fft', Rfft2d(blocksize=blocksize, interleaving=False))

        # Learnable Paramters
        self.alpha_1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.alpha_3 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.p_1_tilde = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.p_2_tilde = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.p_3_tilde = nn.Parameter(torch.tensor(0.), requires_grad=True)

        # Sensitivity table half size due to rfft
        tsize  = (blocksize, blocksize//2+1)
        self.T_tilde = nn.Parameter(torch.zeros(tsize), requires_grad=True)
        
        # Phase weights
        self.W_tilde = nn.Parameter(torch.zeros(tsize), requires_grad=True)

    @property
    def T(self):
        return torch.exp(self.T_tilde)

    @property
    def W(self):
        return torch.exp(self.W_tilde)

    @property
    def p_1(self):
        return 1. + torch.exp(self.p_1_tilde)

    @property
    def p_2(self):
        return 1. + torch.exp(self.p_2_tilde)
    
    @property
    def p_3(self):
        return 1. + torch.exp(self.p_3_tilde)

    def forward(self, img1, img2):
        # Convert to frequency space
        C0 = self.fft(img1)
        C1 = self.fft(img2)

        N, K, H, W, _ = C0.shape
        
        # Retrieve amplitudes by taking the norm of the complex numbers
        #  complex numbers are basically vectors
        C  = torch.norm(C0 + EPS, dim=4)
        C_ = torch.norm(C1 + EPS, dim=4)
        
        # Block intensities
        C0_00 =  C[:, :, 0, 0].view(N, K, 1, 1)
        C1_00 = C_[:, :, 0, 0].view(N, K, 1, 1)
        
        # Calculate average intensity of the images
        C0_bar = torch.mean(C0_00, dim=(1, 2, 3))
        C1_bar = torch.mean(C1_00, dim=(1, 2, 3))
        
        # Luminance normalized blocks
        C0_tilde = C  / (C0_00 + EPS)
        C1_tilde = C_ / (C1_00 + EPS)
        
        # Normalized block intensities
        C0_00_tilde = C0_00 / (C0_bar.view(N, 1, 1, 1) + EPS)
        C1_00_tilde = C1_00 / (C1_bar.view(N, 1, 1, 1) + EPS)
        
        l1 = self.alpha_1 * self._luminance_loss(C0_bar, C1_bar)
        
        l2 = self.alpha_2 * self._block_luminance_loss(C0_00_tilde,
                                                       C1_00_tilde, N)
        l3 = self.alpha_3 * self._frequency_loss(C0_tilde,
                                                 C1_tilde,
                                                 (N, K, H, W))
        percep_dist = l1 + l2 + l3
        
        # Phase part - this is basically just copied from the reference
        #   implementation
        # get phases
        c0_phase = torch.atan2(C0[:,:,:,:,1], C0[:,:,:,:,0] + EPS) 
        c1_phase = torch.atan2(C1[:,:,:,:,1], C1[:,:,:,:,0] + EPS)
        
        # angular distance
        phase_dist = torch.acos(torch.cos(c0_phase - c1_phase)*(1 - EPS*10**3)) * self.W
        phase_dist = torch.sum(phase_dist, dim=(1,2,3))
        
        # perceptual distance
        distance = percep_dist + phase_dist
        return distance

    def _luminance_loss(self, C, C_):
        return (torch.abs(C - C_) + EPS) ** self.p_1

    def _block_luminance_loss(self, C, C_, N):
        # Minkowski distance
        inner = (torch.abs(C - C_) + EPS) ** self.p_2
        return torch.sum(inner, dim=(1,2,3)) ** (1. / self.p_2)

    def _frequency_loss(self, C, C_, dims):
        N, K, H, W = dims
        
        T = self.T.view(1, 1, H, W).expand(N, K, H, W)

        # compute difference
        diff = torch.abs(C - C_)
        # Weight the difference by the sensitivity table
        w_diff = (diff / T) + EPS
        # Compute minkowski distance
        return torch.sum(w_diff ** self.p_3, dim=(1, 2, 3)) ** (1. / self.p_3)

class RGB2YCbCr(nn.Module):
    # Not my code , credit: ?
    def __init__(self):
        super().__init__()
        transf = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).transpose(0, 1)
        self.transform = nn.Parameter(transf, requires_grad=False)
        bias = torch.tensor([0, 0.5, 0.5])
        self.bias = nn.Parameter(bias, requires_grad=False)
    
    def forward(self, rgb):
        N, C, H, W = rgb.shape
        assert C == 3
        rgb = rgb.transpose(1,3)
        cbcr = torch.matmul(rgb, self.transform)
        cbcr += self.bias
        return cbcr.transpose(1,3)

class MultiChannelCost(nn.Module):

    def __init__(self, blocksize=8):
        super(MultiChannelCost, self).__init__()

        self.add_module('to_YCbCr', RGB2YCbCr())
        self.add_module('ly', PerceptualCost(blocksize=blocksize))
        self.add_module('lcb', PerceptualCost(blocksize=blocksize))
        self.add_module('lcr', PerceptualCost(blocksize=blocksize))

        self.lambdas = nn.Parameter(torch.zeros(3))

    @property
    def l(self):
        return F.softmax(self.lambdas, dim=0)

    def forward(self, img1, img2):
        ins = self.to_YCbCr(img1)
        tar = self.to_YCbCr(img2)
        
        ly  = self.ly(ins[:, [0], :, :], tar[:, [0], :, :])
        lcb = self.lcb(ins[:, [1], :, :], tar[:, [1], :, :])
        lcr = self.lcr(ins[:, [2], :, :], tar[:, [2], :, :])
        
        l = self.l
        
        return ly * l[0] + lcb * l[1] + lcr * l[2]

class WeightedSigmoidBCE(nn.Module):

    def __init__(self):
        super(WeightedSigmoidBCE, self).__init__()
        
        self.w_tilde = torch.Tensor([0.])
        self.w_tilde = nn.Parameter(self.w_tilde)
        self.zero    = torch.Tensor([0.])
        self.loss    = torch.nn.BCEWithLogitsLoss()

    @property
    def w(self):
        return torch.exp(self.w_tilde)

    def G(self, d0, d1):
        """Compute Ranking Probability"""
        # I guess it is safe to assume d0 >= 0 & d1 >= 0
        #  since the cost is metric
        normed_diff = (d1 - d0) / (d1 + d0 + EPS)
        return self.w * normed_diff

    def forward(self, d0, d1, judge):
        g_logits = self.G(d0, d1)
        return self.loss(g_logits.view(-1, 1), judge)

class CostTrainingHarness:

    def __init__(self, cuda=True):
        self.cuda = cuda
        self.cost = MultiChannelCost()
        self.loss = WeightedSigmoidBCE()
    
        self.cost = self.cost.cuda() if cuda else self.cost
        self.loss = self.loss.cuda() if cuda else self.loss

        self.parameters  = list(self.cost.parameters())
        self.parameters += list(self.loss.parameters()) 
        
        self.opt = torch.optim.Adam(
            self.parameters,
            lr=0.00001,
            weight_decay=0. # Tweak this
        )

        # Utils
        self.total_its = 1
        self.vis_nstep = 20
        self.vis = Visualizer("./checkpoint/perceptual_cost/loss_function.png")
    
    def compute_accuracy(self,d0,d1,judge):
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        acc_r = d1_lt_d0*judge_per + (1-d1_lt_d0)*(1-judge_per)
        return np.mean(acc_r)

    def train(self, dataloader, epochs=100):
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                # Start by resetting grads
                self.opt.zero_grad()

                # Unpack data
                ref   = data['ref']
                p0    = data['p0']
                p1    = data['p1']
                judge = data['judge']
                
                ref   = ref.cuda()   if self.cuda else ref
                p0    = p0.cuda()    if self.cuda else p0
                p1    = p1.cuda()    if self.cuda else p1
                judge = judge.cuda() if self.cuda else judge
               
                # Compute distances
                d0 = self.cost(ref, p0)
                d1 = self.cost(ref, p1)
               
                # Compute Accuracy
                acc = self.compute_accuracy(d0, d1, judge)

                # Compute loss
                loss = self.loss(d0, d1, judge)
                
                # Backprop
                loss.backward()
                self.opt.step()

                #.format(epoch, i, loss) Print - should be writing to file
                print("Epoch : {} iteration : {}  -- Loss : {} -- Accurcay : {}".format(epoch,
                                                                       i, loss, acc))

                self.total_its += 1
                # Utils
                if i % self.vis_nstep == 0:
                    self.vis.append_data({
                        "iteration": self.total_its,
                        "loss"     : [float(loss.detach().numpy())]
                    })
                    self.vis.plot_and_save("iteration", "loss")

if __name__ == '__main__':
    from torchviz import make_dot

    img1 = torch.randn((1, 3, 64, 64))
    img2 = torch.randn((1, 3, 64, 64))
    img3 = torch.randn((1, 3, 64, 64))

    p = MultiChannelCost()
    
    w = make_dot(p(img1, img2), params=dict(p.named_parameters()))
    print(w)
    w.view()
