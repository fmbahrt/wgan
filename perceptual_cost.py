import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10

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
        super().__init__() # call super constructor
        
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
    
class PerceptualCost(nn.Module):

    def __init__(self, blocksize=8):
        super(PerceptualCost, self).__init__()

        self.add_module('fft', Rfft2d(blocksize=blocksize, interleaving=False))

        # Learnable Paramters
        self.alpha_1 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.alpha_3 = nn.Parameter(torch.tensor(1.), requires_grad=True)

        self.p_1 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.p_2_tilde = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.p_3_tilde = nn.Parameter(torch.tensor(1.), requires_grad=True)

        #  Sensitivity table half size due to rfft
        tsize  = (blocksize, blocksize//2+1)
        self.T_tilde = nn.Parameter(torch.ones(tsize), requires_grad=True)

        # Phase weights
        self.W_tilde = nn.Parameter(torch.ones(tsize), requires_grad=True)

    @property
    def T(self):
        return torch.exp(self.T_tilde)

    @property
    def W(self):
        return torch.exp(self.W_tilde)

    @property
    def p_2(self):
        return 1. - torch.exp(self.p_2_tilde)
    
    @property
    def p_3(self):
        return 1. - torch.exp(self.p_3_tilde)

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
        return torch.abs(C - C_) ** self.p_1

    def _block_luminance_loss(self, C, C_, N):
        # Minkowski distance
        inner = torch.abs(C - C_) ** self.p_2
        return torch.sum(inner, dim=(1,2,3)) ** (1. / self.p_2)

    def _frequency_loss(self, C, C_, dims):
        N, K, H, W = dims

        T = self.T.view(1, 1, H, W).expand(N, K, H, W)

        # compute difference
        diff = torch.abs(C - C_)

        # Weight the difference by the sensitivity table
        w_diff = diff / T

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

        self._lambdas = nn.Parameter(torch.zeros(3), requires_grad=True)

    @property
    def l(self):
        return F.softmax(self._lambdas, dim=0)

    def forward(self, img1, img2):
        ins = self.to_YCbCr(img1)
        tar = self.to_YCbCr(img2)

        ly  = self.ly(ins[:, [0], :, :], tar[:, [0], :, :])
        lcb = self.lcb(ins[:, [1], :, :], tar[:, [1], :, :])
        lcr = self.lcr(ins[:, [2], :, :], tar[:, [2], :, :])

        l = self.l

        return ly * l[0] + lcb * l[1] + lcr * l[2]

class WeightedSigmoidBCE(nn.Module):

    def __init__(self, blocksize=8):
        super(WeightedSigmoidBCE, self).__init__()
        
        self.w_tilde = torch.Tensor([1.])
        self.w_tilde = nn.Parameter(self.w_tilde)
        self.half    = torch.Tensor(.5)
        self.loss    = torch.nn.BCELoss()

    @property
    def w(self):
        return torch.exp(self.w_tilde)

    def G(d0, d1):
        """Compute Ranking Probability"""
        # I guess it is safe to assume d0 > 0 & d1 > 0
        #  since the cost is metric

        normed_diff = (d1 - d0) / d1 + d0
        normed_diff = torch.where((d0+d1) > 0., normed_diff, self.half)
        w_sigmoid   = torch.sigmoid(self.w * normed_diff)
        return w_sigmoid

    def forward(d0, d1, judge):
        return self.loss(self.G(d0, d1), judge)

class CostTrainingHarness:

    def __init__(self, cost, gpu=True):
        self.gpu  = gpu
        self.cost = cost

        self.opt = torch.optim.Adam(
            self.cost.parameters(),
            lr=1e-3,
            weight_decay=1e-3 # Tweak this
        )

    def train(self, dataloader, epochs=100):
        return

if __name__ == '__main__':
    img1 = torch.randn((2, 3, 64, 64))
    img2 = torch.randn((2, 3, 64, 64))

    p = MultiChannelCost()
    
    print(p.forward(img1, img2))

