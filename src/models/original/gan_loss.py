import torch
import torch.nn as nn
from torch.autograd import Variable

class GANLoss(nn.Module):
    """ 
    Reward-Refined NLLLoss Function for adversarial reinforcement training of generator
    """
    def __init__(self, use_cuda, **kwargs):
        self.use_cuda = use_cuda
        super(GANLoss, self).__init__(**kwargs)

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C) - torch Variable
            target: (N,) - torch Variable
            reward: (N,) - torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        indices = target.data.view((-1, 1))
        if self.use_cuda:
            one_hot = one_hot.cuda()   
            indices = indices.cuda()
        # write 1 into all positions specified by target in the 1st dim
        one_hot.scatter_(1, target.data.view((-1, 1)), 1) 
        one_hot = Variable(one_hot.type(torch.ByteTensor)) # sets the type, so it can be used in masked_select
        if self.use_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward # why does a greater reward = greater loss? This should be opposite the case.
        loss = -torch.sum(loss)
        return loss

