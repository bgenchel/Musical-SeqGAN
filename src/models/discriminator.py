"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""

import torch
import toch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, 
            num_filters, output_dim, dropout=0.0, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.fc1 = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(sum(num_filters), output_dim)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            # note to self, look up why this range is chosen
            param.data.uniform(-0.05, 0.05)

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        # batch_size * 1 * seq_len * embed_dim
        embedded = self.embedder(x).unsqueeze(1)
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(embedded)).sqeeze(3) for conv in self.conv_layers]
        # [batch_size * num_filter]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1) # batch_size * num_filters_sum
        pred = self.fc1(pred)
        pred = F.sigmoid(pred) * F.relu(pred) + (1. - F.sigmoid(pred)) * pred
        pred = self.softmax(self.fc2(self.dropout(pred)))
        return pred
