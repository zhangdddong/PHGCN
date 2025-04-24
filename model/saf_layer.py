import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # self.query = nn.Linear(input_dim, hidden_dim)
        # self.key = nn.Linear(input_dim, hidden_dim)
        # self.value = nn.Linear(input_dim, hidden_dim)
        # self.scale = 1.0 / (hidden_dim ** 0.5)

        # linear weight
        self.fusion_linear = nn.Linear(input_dim * 2, hidden_dim)

    def forward(self, a, b):
        # # Compute query, key, and value
        # query = self.query(a)  # Shape: (batch_size, hidden_dim)
        # key = self.key(b)      # Shape: (batch_size, hidden_dim)
        # value = self.value(b)  # Shape: (batch_size, hidden_dim)
        # # Compute attention scores
        # scores = torch.matmul(query, key.transpose(-2, -1))
        # scores *= self.scale  # Shape: (batch_size, 1)
        # # Apply softmax to get attention weights
        # attn_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, 1)
        # # Compute the weighted sum of values
        # context = torch.matmul(attn_weights, value)  # Shape: (batch_size, hidden_dim)

        # linear weight
        context = self.fusion_linear(torch.concat([a, b], dim=-1))

        return context


class SAF(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SAF, self).__init__()

        self.x_fc = nn.Linear(input_size, hidden_size)
        self.y_fc = nn.Linear(input_size, hidden_size)

        self.dropout = nn.Dropout(0.8)

        self.kernel_size = 3
        self.stride = 3

        self.attention_layer = Attention(hidden_size * 2, hidden_size)

    def forward(self, x, y):

        # Expand Stage
        x_sp = self.x_fc(x)
        y_sp = self.y_fc(y)
        fusion = x_sp * y_sp
        fusion = torch.cat([fusion, x_sp, y_sp], dim=1)
        fusion = self.dropout(fusion)

        # Squeeze Stage
        fusion = fusion.unsqueeze(0)
        fusion = self.sum_pooling(fusion)
        fusion = F.normalize(fusion, p=2, dim=1).squeeze(0)

        # attention fusion
        fusion_x = torch.cat([fusion, x], dim=1)
        fusion_y = torch.cat([fusion, y], dim=1)
        fusion = self.attention_layer(fusion_x, fusion_y)

        return fusion

    def sum_pooling(self, input_vector):
        if self.stride is None:
            self.stride = self.kernel_size
        # Apply average pooling
        avg_pooled = F.avg_pool1d(input_vector, kernel_size=self.kernel_size, stride=self.stride)
        # Scale the average pooled values by the kernel size
        sum_pooled = avg_pooled * (self.kernel_size * self.kernel_size)
        return sum_pooled
