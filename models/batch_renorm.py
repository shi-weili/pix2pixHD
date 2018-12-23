import torch
import torch.nn as nn

class BatchRenorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.99,
                 renorm_startpoint=5000, d_max=5, r_max=3):
        super(BatchRenorm2d, self).__init__()

        self.eps = eps
        self.momentum = momentum
        self.renorm_startpoint = renorm_startpoint
        self.d_max = d_max
        self.r_max = r_max

        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_sigma', torch.zeros(1, num_features, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_sigma.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()
        self.num_batches_tracked.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        if self.training:
            self.num_batches_tracked += 1

            mean_in = x.mean(-1, keepdim=True)
            var_in = x.var(-1, keepdim=True)
            temp = var_in + mean_in ** 2

            mean = mean_in.mean(0, keepdim=True)
            sigma = (temp.mean(0, keepdim=True) - mean ** 2).sqrt()

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean.data)
            self.running_sigma.mul_(self.momentum)
            self.running_sigma.add_((1 - self.momentum) * sigma.data)

            if self.num_batches_tracked > self.renorm_startpoint:
                d = (mean.detach() - self.running_mean) / self.running_sigma
                d_max = self.d_max * max(self.num_batches_tracked - self.renorm_startpoint, 0) / 25000
                d.clamp_(-d_max, d_max)

                r = sigma.detach() / self.running_sigma
                r_max = 1 + (self.r_max - 1) * max(self.num_batches_tracked - self.renorm_startpoint, 0) / 40000
                r.clamp_(1/r_max, r_max)

            else:
                d = 0
                r = 1
        else:
            mean = torch.autograd.Variable(self.running_mean)
            sigma = torch.autograd.Variable(self.running_sigma)

            d = 0
            r = 1

        x = (x - mean) / (sigma + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
