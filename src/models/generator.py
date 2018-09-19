"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, use_cuda=True, **kwargs):
        super(Generator, self).__init__(**kwargs)
        # the number of discrete values an input can take on
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.num_layers = 1

        self.embedder = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
        return

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        cell = Variable(torch.zeros((self.num_layers, batch_size, self.hidden_dim)))
        if self.use_cuda and torch.cuda.is_available():
            hidden, cell = hidden.cuda(), cell.cuda()
        return hidden, cell

    def forward(self, x):
        embedded = self.embedder(x)
        h0, c0 = self.init_hidden(x.size[0])
        lstm_out, _ = self.lstm(embedded, (h0, c0))
        softmax = self.softmax(self.fc(lstm_out.view(-1, self.hidden_dim)))
        return softmax
