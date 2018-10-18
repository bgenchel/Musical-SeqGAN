"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import argparse
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from gan_loss import GANLoss
# things I haven't really looked through yet but had to copy in due to time constraint
from target_lstm import TargetLSTM
from data_iter import GenDataIter, DisDataIter

parser = argparse.ArgumentParser(description="Training Parameter")
parser.add_argument('--no_cuda', action='store_true', 
                    help="don't use CUDA, even if it is available.")
args = parser.parse_args()

args.cuda = False
if torch.cuda.is_available() and (not args.no_cuda):
    torch.cuda.set_device(0) # just default it for now, maybe change later
    args.cuda = True

# Basic Training Paramters
SEED = 88 # seems like quite a long seed
BATCH_SIZE = 64
TOTAL_BATCH = 200 # ??
GENERATED_NUM = 10000 # ??
POSITIVE_FILE = 'real.data' # not sure what is meant by 'positive'
NEGATIVE_FILE = 'gene.data' # not sure what is meant by 'negative'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 5000 # ??
GEN_PRETRAIN_EPOCHS = 120 # ??
DSCR_PRETRAIN_DATA_GENS = 5
DSCR_PRETRAIN_EPOCHS = 3
# GEN_PRETRAIN_EPOCHS = 0
# DSCR_PRETRAIN_DATA_GENS = 0
# DSCR_PRETRAIN_EPOCHS = 0

# Generator Params
gen_embed_dim = 32
gen_hidden_dim = 32
gen_seq_len = 20

# Discriminator Parameters
dscr_embed_dim = 64
dscr_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dscr_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dscr_dropout = 0.75
# why not just have one class that indicates probability of being real with a sigmoid
# output? The probability between two classes should just sum to 1 anyways
dscr_num_classes = 2 


def generate_samples(model, batch_size, num_samples, output_file):
    samples = []
    for _ in range(int(num_samples / batch_size)):
        # why cpu?
        sample_batch = model.sample(batch_size, gen_seq_len).cpu().data.numpy().tolist()
        samples.extend(sample_batch)
        with open(output_file, 'w') as fout:
            for sample in samples:
                str_sample = ' '.join([str(s) for s in sample])
                fout.write('%s\n' % str_sample)
    return

def train_epoch(model, data_iter, loss_fn, optimizer):
    total_loss = 0.0
    total_words = 0.0 # ???
    for (data, target) in tqdm(data_iter, desc=' - Training', leave=False):
        data_var = Variable(data)
        target_var = Variable(target)
        if args.cuda:
            data_var, target_var = data_var.cuda(), target_var.cuda()
        target_var = target_var.contiguous().view(-1) # serialize the target into a contiguous vector ?
        pred = model.forward(data_var)
        loss = loss_fn(pred, target_var)
        total_loss += loss.item()
        total_words += data_var.size(0) * data_var.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words) # weird measure ... to return

def eval_epoch(model, data_iter, loss_fn):
    total_loss = 0.0
    total_words = 0.0
    with torch.no_grad():
        for (data, target) in tqdm(data_iter, desc= " - Evaluation", leave=False):
            data_var = Variable(data)
            target_var = Variable(target)
            if args.cuda:
                data_var, target_var = data_var.cuda(), target_var.cuda()
            target_var = target_var.contiguous().view(-1) # serialize the target into a contiguous vector ?
            pred = model.forward(data_var)
            loss = loss_fn(pred, target_var)
            total_loss += loss.item()
            total_words += data_var.size(0) * data_var.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words) # weird measure ... to return

# definitely need to go through this still
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(VOCAB_SIZE, gen_embed_dim, gen_hidden_dim, args.cuda)
    discriminator = Discriminator(VOCAB_SIZE, dscr_embed_dim, dscr_filter_sizes, dscr_num_filters, dscr_num_classes, dscr_dropout)
    target_lstm = TargetLSTM(VOCAB_SIZE, gen_embed_dim, gen_hidden_dim, args.cuda)
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        target_lstm = target_lstm.cuda()

    # Generate toy data using target lstm
    print('Generating data ...')
    generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)
    
    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    print('Pretrain Generator with MLE ...')
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters())
    if args.cuda:
        gen_criterion = gen_criterion.cuda()
    for epoch in range(GEN_PRETRAIN_EPOCHS):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    # Pretrain Discriminator
    print('Pretrain Discriminator ...')
    dscr_criterion = nn.NLLLoss(size_average=False)
    dscr_optimizer = optim.Adam(discriminator.parameters())
    if args.cuda:
        dscr_criterion = dscr_criterion.cuda()
    for i in range(DSCR_PRETRAIN_DATA_GENS):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dscr_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for j in range(DSCR_PRETRAIN_EPOCHS):
            loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)
            print('Data Gen [%d], Epoch [%d], Loss: %f' % (i, j, loss))

    # Adversarial Training 
    print('#'*100)
    print('Start Adeversatial Training...\n')

    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss(use_cuda=args.cuda)
    gen_gan_optm = optim.Adam(generator.parameters())
    if args.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.NLLLoss(size_average=False)
    if args.cuda:
        gen_criterion = gen_criterion.cuda()

    dscr_criterion = nn.NLLLoss(size_average=False)
    dscr_optimizer = optim.Adam(discriminator.parameters())
    if args.cuda:
        dscr_criterion = dscr_criterion.cuda()

    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        samples = generator.sample(BATCH_SIZE, gen_seq_len)
        # construct the input to the genrator, add zeros before samples and delete the last column
        zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
        if samples.is_cuda:
            zeros = zeros.cuda()
        inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
        targets = Variable(samples.data).contiguous().view((-1,))
        # calculate the reward
        rewards = rollout.get_reward(samples, 16, discriminator)
        rewards = Variable(torch.Tensor(rewards))
        rewards = torch.exp(rewards).contiguous().view((-1,))
        if args.cuda:
            rewards = rewards.cuda()
        prob = generator.forward(inputs)
        loss = gen_gan_loss(prob, targets, rewards)
        gen_gan_optm.zero_grad()
        loss.backward()
        gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
            print('Batch [%d] True Loss: %f' % (total_batch, loss))
        rollout.update_params()
        
        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dscr_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(2):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)

if __name__ == '__main__':
    main()
