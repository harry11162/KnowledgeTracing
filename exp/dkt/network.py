import torch
import torch.nn as nn
from lib.loss import BCE_loss_on_skills

class DKT(nn.Module):
    def __init__(self, cfg):
        super(DKT, self).__init__()
        self.cell = nn.LSTMCell(
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
            y.append(y_t)
            h_0, c_0 = self.cell(x_t, (h_0, c_0))

        y = torch.stack(y)

        if self.training:
            return BCE_loss_on_skills(y, skills, target)
        else:
            y = torch.sigmoid(y)
            return y