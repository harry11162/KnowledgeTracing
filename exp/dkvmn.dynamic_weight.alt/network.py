import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.loss import BCE_loss

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
        


class DKVMNcell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, dsize=8):
        super(DKVMNcell, self).__init__()
        self.key_embed = nn.Linear(input_size // 2, hidden_size)
        self.km = nn.Linear(hidden_size, memory_size)
        self.kw2 = nn.Linear(hidden_size, dsize)

        self.summary = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

        self.value_embed = nn.Linear(input_size, hidden_size)
        # self.erase = nn.Linear(hidden_size, hidden_size)
        # self.add = nn.Linear(hidden_size, hidden_size)
        self.erase = DynamicLinear(hidden_size, hidden_size, dsize)
        self.add = DynamicLinear(hidden_size, hidden_size, dsize)

        self.hidden_size = hidden_size

    def forward(self, x, memory):
        N, C2 = x.size()
        C = C2 // 2
        q = x.reshape(N, 2, C).sum(dim=1)

        k = self.key_embed(q)
        w = F.softmax(self.km(k), dim=-1)  # (N, M)
        w2 = F.softmax(self.kw2(k), dim=-1)

        # memory (N, hidden_size, M)
        r = (memory * w.unsqueeze(1)).sum(dim=2)

        kr = torch.cat([k, r], dim=1)
        f = torch.tanh(self.summary(kr))
        y = self.output(f)

        v = self.value_embed(x)
        e = torch.sigmoid(self.erase(v, w2))
        a = torch.tanh(self.add(v, w2))
        memory = memory * (1 - w.unsqueeze(dim=1) * e.unsqueeze(dim=2))
        memory = memory + w.unsqueeze(dim=1) * a.unsqueeze(dim=2)

        return y, memory



class DKT(nn.Module):
    def __init__(self, cfg):
        super(DKT, self).__init__()
        self.cell = DKVMNcell(
            input_size=cfg.data.input_size,
            hidden_size=cfg.model.hidden_size,
            memory_size=cfg.model.memory_size,
        )
        self.input_size = cfg.data.input_size
        self.hidden_size = cfg.model.hidden_size
        self.memory_size = cfg.model.memory_size
    
    def forward(self, x, skills, target=None):
        L, N, C2 = x.size()

        y = []
        m_0 = x.new_zeros((N, self.hidden_size, self.memory_size))
        for x_t in x:
            y_t, m_0 = self.cell(x_t, m_0)
            y.append(y_t)

        y = torch.stack(y)

        if self.training:
            return BCE_loss(y, target)
        else:
            y = torch.sigmoid(y)
            return y
