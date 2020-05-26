import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.loss import BCE_loss_on_skills

# class DynamicLinear(nn.Module):
#     def __init__(self, in_features, out_features, dsize):
#         super(DynamicLinear, self).__init__()
#         self.weights = nn.Parameter(torch.Tensor(dsize, out_features, in_features))
#         self.biases = nn.Parameter(torch.Tensor(dsize, out_features))
#         self.reset_parameters()
# 
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in)
#         nn.init.uniform_(self.biases, -bound, bound)
# 
#     def forward(self, input, w):
#         weight = self.weights[None, :,:,:] * w[:,:, None, None]
#         weight = weight.sum(dim=1)
#         bias = self.biases[None, :,:] * w[:,:, None]
#         bias = bias.sum(dim=1)
#         input = torch.matmul(input[:, None, :], weight.permute(0, 2, 1))
#         input = input.squeeze(1) + bias
#         return input
        


class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMcell, self).__init__()
        self.ii = nn.Linear(input_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)
        self.If = nn.Linear(input_size, hidden_size)
        self.hf = nn.Linear(hidden_size, hidden_size)
        self.ig = nn.Linear(input_size, hidden_size)
        self.hg = nn.Linear(hidden_size, hidden_size)
        self.io = nn.Linear(input_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, x, h0c0):
        h, c = h0c0

        i = torch.sigmoid(self.ii(x) + self.hi(h))
        f = torch.sigmoid(self.If(x) + self.hf(h))
        g = torch.tanh(self.ig(x) + self.hg(h))
        o = torch.sigmoid(self.io(x) + self.ho(h))

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c



class DKT(nn.Module):
    def __init__(self, cfg):
        super(DKT, self).__init__()
        self.cell = LSTMcell(
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
        for x_t in x:
            y_t = self.output_layer(c_0)
            if not self.training:
                y.append(y_t)
            h_0, c_0 = self.cell(x_t, (h_0, c_0))
        if self.training:
            y.append(y_t)

        y = torch.stack(y)

        if self.training:
            return BCE_loss_on_skills(y, skills, target)
        else:
            y = torch.sigmoid(y)
            return y
