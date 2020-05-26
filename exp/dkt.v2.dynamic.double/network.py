import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.loss import BCE_loss_on_skills

class DynamicLinear(nn.Module):
    def __init__(self, in_features, out_features, dsize):
        super(DynamicLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dsize, out_features, in_features))
        self.biases = nn.Parameter(torch.Tensor(dsize, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, input, w):
        weight = self.weights[None, :,:,:] * w[:,:, None, None]
        weight = weight.sum(dim=1)
        bias = self.biases[None, :,:] * w[:,:, None]
        bias = bias.sum(dim=1)
        input = torch.matmul(input[:, None, :], weight.permute(0, 2, 1))
        input = input.squeeze(1) + bias
        return input
        


class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size, dsize=8):
        super(LSTMcell, self).__init__()
        self.ii = DynamicLinear(input_size, hidden_size, dsize=dsize)
        self.hi = DynamicLinear(hidden_size, hidden_size, dsize=dsize)
        self.If = DynamicLinear(input_size, hidden_size, dsize=dsize)
        self.hf = DynamicLinear(hidden_size, hidden_size, dsize=dsize)
        self.ig = DynamicLinear(input_size, hidden_size, dsize=dsize)
        self.hg = DynamicLinear(hidden_size, hidden_size, dsize=dsize)
        self.io = DynamicLinear(input_size, hidden_size, dsize=dsize)
        self.ho = DynamicLinear(hidden_size, hidden_size, dsize=dsize)

        self.pi = nn.Linear(hidden_size, dsize)
        self.hidden_size = hidden_size

    def forward(self, x, h0c0s):
        h, c, s = h0c0s

        pi = torch.sigmoid(self.pi(s))

        i = torch.sigmoid(self.ii(x, pi) + self.hi(h, pi))
        f = torch.sigmoid(self.If(x, pi) + self.hf(h, pi))
        g = torch.tanh(self.ig(x, pi) + self.hg(h, pi))
        o = torch.sigmoid(self.io(x, pi) + self.ho(h, pi))

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c



class DKT(nn.Module):
    def __init__(self, cfg, dsize=8):
        super(DKT, self).__init__()
        self.cell = LSTMcell(
            input_size=cfg.data.input_size,
            hidden_size=cfg.model.hidden_size,
            dsize=dsize,
        )
        self.stu = nn.LSTMCell(
            input_size=cfg.data.input_size,
            hidden_size=cfg.model.hidden_size,
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(cfg.model.hidden_size, cfg.data.input_size),
        )
        self.input_size = cfg.data.input_size
        self.hidden_size = cfg.model.hidden_size
    
    def forward(self, x, skills, target=None):
        L, N, C2 = x.size()

        y = []
        h_0 = x.new_zeros((N, self.hidden_size))
        c_0 = x.new_zeros((N, self.hidden_size))
        s_h = x.new_zeros((N, self.hidden_size))
        s_c = x.new_zeros((N, self.hidden_size))
        for x_t in x:
            y_t = self.output_layer(c_0)
            y.append(y_t)
            s_h, s_c = self.stu(x_t, (s_h, s_c))
            h_0, c_0 = self.cell(x_t, (h_0, c_0, s_c))

        y = torch.stack(y)

        if self.training:
            return BCE_loss_on_skills(y, skills, target)
        else:
            y = torch.sigmoid(y)
            return y
