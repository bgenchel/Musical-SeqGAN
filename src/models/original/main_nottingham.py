"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import argparse
import json
import math
import numpy as np
import random
import os
import os.path as op
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from gan_loss import GANLoss
# from target_lstm import TargetLSTM
from data_iter import GenDataIter, DisDataIter

sys.path.append('../../data')
from parsing.datasets import NottinghamDataset

parser = argparse.ArgumentParser(description="Training Parameter")
parser.add_argument('-tt', '--train_type', choices=("full_sequence", "next_step"), 
                    default="full_sequence", help="how to train the network")
parser.add_argument('-glr', '--gen_learning_rate', default=1e-2, type=float, help="learning rate for generator")
parser.add_argument('-dlr', '--dscr_learning_rate', default=1e-3, type=float, help="learning rate for discriminator")
parser.add_argument('-nc', '--no_cuda', action='store_true', 
                    help="don't use CUDA, even if it is available.")
args = parser.parse_args()

args.cuda = False
if torch.cuda.is_available() and (not args.no_cuda):
    torch.cuda.set_device(0) # just default it for now, maybe change later
    args.cuda = True

# Basic Training Paramters
SEED = 88 # seems like quite a long seed, how was this chosen?
BATCH_SIZE = 64
TOTAL_BATCH = 200 # ??
GENERATED_NUM = 10000 # ??
POSITIVE_FILE = 'real.data' # not sure what is meant by 'positive'
NEGATIVE_FILE = 'gene.data' # not sure what is meant by 'negative'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 89
# GEN_PRETRAIN_EPOCHS = 120 # ??
GEN_PRETRAIN_EPOCHS = 200 # ??
DSCR_PRETRAIN_DATA_GENS = 5
DSCR_PRETRAIN_EPOCHS = 3

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


def train_epoch(model, data_iter, loss_fn, optimizer, train_type):
    total_loss = 0.0
    total_words = 0.0 # ???
    for (data, target) in tqdm(data_iter, desc=' - Training', leave=False):
        data_var = Variable(data)
        target_var = Variable(target)
        if args.cuda:
            data_var, target_var = data_var.cuda(), target_var.cuda()
        target_var = target_var.contiguous().view(-1) # serialize the target into a contiguous vector ?
        pred = model.forward(data_var)
        if train_type == "full_sequence":
            pred = pred.view(-1, pred.size()[-1])
        elif train_type == "next_step":
            pred = pred[:, -1, :]
        loss = loss_fn(pred, target_var)
        total_loss += loss.item()
        total_words += data_var.size(0) * data_var.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    return math.exp(total_loss / total_words) # weird measure ... to return

# definitely need to go through this still
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', train_type=args.train_type, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Networks
    generator = Generator(VOCAB_SIZE, gen_embed_dim, gen_hidden_dim, args.cuda)
    discriminator = Discriminator(VOCAB_SIZE, dscr_embed_dim, dscr_filter_sizes, dscr_num_filters, dscr_num_classes, dscr_dropout)
    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # # Generate toy data using target lstm
    # print('Generating data ...')
    # generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE

    # Pretrain Generator using MLE
    print('Pretrain Generator with MLE ...')
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
    if args.cuda:
        gen_criterion = gen_criterion.cuda()
    for epoch in range(GEN_PRETRAIN_EPOCHS):
        loss = train_epoch(generator, dataloader, gen_criterion, gen_optimizer, args.train_type)
        print('Epoch [%d] Training Loss: %f'% (epoch, loss))
        # generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        # eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        # loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        # print('Epoch [%d] True Loss: %f' % (epoch, loss))

    run_dir = op.join("runs", datetime.now().strftime('%b%d-%y_%H:%M:%S'))
    if not op.exists(run_dir):
        os.makedirs(run_dir)

    model_inputs = {'vocab_size': VOCAB_SIZE,
                    'embed_dim': gen_embed_dim,
                    'hidden_dim': gen_hidden_dim,
                    'use_cuda': False}
    json.dump(model_inputs, open(op.join(run_dir, 'model_inputs.json'), 'w'), indent=4)
    torch.save(generator.state_dict(), op.join(run_dir, 'generator_state.pt'))

    # Pretrain Discriminator
    # print('Pretrain Discriminator ...')
    # dscr_criterion = nn.NLLLoss(size_average=False)
    # dscr_optimizer = optim.Adam(discriminator.parameters())
    # if args.cuda:
    #     dscr_criterion = dscr_criterion.cuda()
    # for i in range(DSCR_PRETRAIN_DATA_GENS):
    #     generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
    #     dscr_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
    #     for j in range(DSCR_PRETRAIN_EPOCHS):
    #         loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)
    #         print('Data Gen [%d], Epoch [%d], Loss: %f' % (i, j, loss))

    # # Adversarial Training 
    # print('#'*100)
    # print('Start Adeversatial Training...\n')

    # rollout = Rollout(generator, 0.8)

    # gen_gan_loss = GANLoss(use_cuda=args.cuda)
    # gen_gan_optm = optim.Adam(generator.parameters())
    # if args.cuda:
    #     gen_gan_loss = gen_gan_loss.cuda()
    # gen_criterion = nn.NLLLoss(size_average=False)
    # if args.cuda:
    #     gen_criterion = gen_criterion.cuda()

    # dscr_criterion = nn.NLLLoss(size_average=False)
    # dscr_optimizer = optim.Adam(discriminator.parameters())
    # if args.cuda:
    #     dscr_criterion = dscr_criterion.cuda()

    # for total_batch in range(TOTAL_BATCH):
    #     ## Train the generator for one step
    #     samples = generator.sample(BATCH_SIZE, gen_seq_len)
    #     # construct the input to the genrator, add zeros before samples and delete the last column
    #     zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
    #     if samples.is_cuda:
    #         zeros = zeros.cuda()
    #     inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
    #     targets = Variable(samples.data).contiguous().view((-1,))
    #     # calculate the reward
    #     rewards = rollout.get_reward(samples, 16, discriminator)
    #     rewards = Variable(torch.Tensor(rewards))
    #     rewards = torch.exp(rewards).contiguous().view((-1,))
    #     if args.cuda:
    #         rewards = rewards.cuda()
    #     prob = generator.forward(inputs)
    #     loss = gen_gan_loss(prob, targets, rewards)
    #     gen_gan_optm.zero_grad()
    #     loss.backward()
    #     gen_gan_optm.step()

    #     if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
    #         generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
    #         eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
    #         loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
    #         print('Batch [%d] True Loss: %f' % (total_batch, loss))
    #     rollout.update_params()
        
    #     for _ in range(4):
    #         generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
    #         dscr_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
    #         for _ in range(2):
    #             loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer)

if __name__ == '__main__':
    main()
