import torch
import torch.nn as nn
import torch.nn.functional as F

class mcLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels

        self.ih = nn.Linear(self.input_size,
                            (3 + self.num_channels) * self.hidden_size)
        self.hh = nn.Linear(self.hidden_size * self.num_channels,
                            (3 + self.num_channels) * self.hidden_size)

    def forward(self, input, hidden):
        hxs, cx = hidden[:-1], hidden[-1]
        gates = self.ih(input) + self.hh(torch.cat(hxs, dim=-1))

        chunked_gates = gates.chunk(3 + self.num_channels, dim=-1)
        (ingate, forgetgate, cellgate), outgates = chunked_gates[:3], chunked_gates[3:]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgates = [F.sigmoid(og) for og in outgates]

        cy = (forgetgate * cx) + (ingate * cellgate)
        cy_tanh = F.tanh(cy)
        hys = [og * cy_tanh for og in outgates]
        
        hidden = hys + [cy]

        return hidden  # outputs = hidden[:-1]
