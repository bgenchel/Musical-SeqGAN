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
from data_iter import GenDataset, DscrDataset

sys.path.append('../../data')
from parsing.datasets import NottinghamMLEDataset, NottinghamNonOverlapDataset
from loading.dataloaders import SplitDataLoader

parser = argparse.ArgumentParser(description="Training Parameter")
parser.add_argument('-tt', '--train_type', choices=("full_sequence", "next_step"), 
                    default="full_sequence", help="how to train the network")
parser.add_argument('-glr', '--gen_learning_rate', default=1e-3, type=float, help="learning rate for generator")
parser.add_argument('-dlr', '--dscr_learning_rate', default=1e-3, type=float, help="learning rate for discriminator")
parser.add_argument('-fpt', '--force_pretrain', default=False, action='store_true', 
                    help="force pretraining of generator and discriminator, instead of loading from cache.")
parser.add_argument('-nc', '--no_cuda', action='store_true', help="don't use CUDA, even if it is available.")
args = parser.parse_args()

args.cuda = False
if torch.cuda.is_available() and (not args.no_cuda):
    torch.cuda.set_device(0) # just default it for now, maybe change later
    args.cuda = True

# General Training Paramters
SEED = 88 # for the randomizer
BATCH_SIZE = 64
GAN_TRAIN_EPOCHS = 200 # number of adversarial training epochs
GENERATED_NUM = 10000 # number of samples for the generator to generate in order to train the discriminator
VOCAB_SIZE = 89

# Pretraining Paramters
GEN_PRETRAIN_EPOCHS = 120 # original val: 120
DSCR_PRETRAIN_DATA_GENS = 5
DSCR_PRETRAIN_EPOCHS = 3

# Adversarial Training Params
NUM_ROLLOUTS = 16
G_STEPS = 1
D_DATA_GENS = 4
D_STEPS = 2

# Paths
REAL_FILE = op.join('temp_data', 'real.data')
GEN_FILE = op.join('temp_data', 'generated.data')
PT_GEN_MODEL_FILE = op.join('pretrained', 'generator.pt')
PT_DSCR_MODEL_FILE = op.join('pretrained', 'discriminator.pt')

# Generator Model Params
gen_embed_dim = 64
gen_hidden_dim = 128 # originally 64
gen_seq_len = 32

# Discriminator Model Parameters
dscr_embed_dim = 128
dscr_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dscr_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dscr_dropout = 0.75
# why not just have one class that indicates probability of being real with a sigmoid
# output? The probability between two classes should just sum to 1 anyways
dscr_num_classes = 2 


def create_generated_data_file(model, num_samples, batch_size, output_file):
    samples = []
    for _ in range(num_samples // batch_size):
        sample_batch = model.sample(batch_size, gen_seq_len).cpu().data.numpy().tolist() # why cpu?
        samples.extend(sample_batch)

    with open(output_file, 'w') as fout:
        for sample in samples:
            str_sample = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % str_sample)
    return


def create_real_data_file(data_iter, output_file):
    samples = []
    data_iter = iter(data_iter)
    for sample_batch in data_iter:
        sample_batch = list(sample_batch.numpy())
        samples.extend(sample_batch)

    with open(output_file, 'w') as fout:
        for sample in samples:
            str_sample = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % str_sample)
    return


def train_epoch(model, data_iter, loss_fn, optimizer, train_type):
    total_loss = 0.0
    total_words = 0.0 # ???
    total_batches = 0.0
    for (data, target) in tqdm(data_iter, desc=' - Training', leave=False):
        data_var = Variable(data)
        target_var = Variable(target)
        if args.cuda and torch.cuda.is_available():
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
        total_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if type(model) == Discriminator:
        # import pdb
        # pdb.set_trace()
        return total_loss / total_batches
    else:
        return total_loss / total_words # weird measure ... to return


def eval_epoch(model, data_iter, loss_fn, train_type):
    total_loss = 0.0
    total_words = 0.0
    with torch.no_grad():
        for (data, target) in tqdm(data_iter, desc= " - Evaluation", leave=False):
            data_var = Variable(data)
            target_var = Variable(target)
            if args.cuda and torch.cuda.is_available():
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

    if type(model) == Discriminator:
        return total_loss / total_batches
    else:
        return total_loss / total_words # weird measure ... to return


# definitely need to go through this still
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    mle_dataset = NottinghamMLEDataset('../../../data/raw/nottingham-midi', seq_len=gen_seq_len, train_type=args.train_type, data_format="nums")
    train_loader, valid_loader = SplitDataLoader(mle_dataset, batch_size=BATCH_SIZE, drop_last=True).split()
    nonoverlap_dataset = NottinghamNonOverlapDataset('../../../data/raw/nottingham-midi', seq_len=gen_seq_len, data_format="nums")
    data_loader = DataLoader(nonoverlap_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Networks
    generator = Generator(VOCAB_SIZE, gen_embed_dim, gen_hidden_dim, args.cuda)
    discriminator = Discriminator(VOCAB_SIZE, dscr_embed_dim, dscr_filter_sizes, dscr_num_filters, dscr_num_classes, dscr_dropout)

    if args.cuda and torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Pretrain Generator using MLE
    if not args.force_pretrain and op.exists(PT_GEN_MODEL_FILE):
        print('Loading Pretrained Generator ...')
        checkpoint = torch.load(PT_GEN_MODEL_FILE)
        generator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained for %d epochs.' % checkpoint['epochs'])
        print('::INFO:: Final Training Loss - %.5f' % checkpoint['train_loss'])
        print('::INFO:: Final Validation Loss - %.5f' % checkpoint['valid_loss'])
    else:
        print('Pretraining Generator with MLE ...')
        gen_criterion = nn.NLLLoss(size_average=False)
        gen_optimizer = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
        min_valid_loss = float('inf')
        if args.cuda and torch.cuda.is_available():
            gen_criterion = gen_criterion.cuda()
        for epoch in range(GEN_PRETRAIN_EPOCHS):
            train_loss = train_epoch(generator, train_loader, gen_criterion, gen_optimizer, args.train_type)
            print('::PRETRAIN GEN:: Epoch [%d] Training Loss: %f'% (epoch, train_loss))
            valid_loss = eval_epoch(generator, valid_loader, gen_criterion, args.train_type)
            print('::PRETRAIN GEN:: Epoch [%d] Validation Loss: %f'% (epoch, valid_loss))
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                print('Caching Pretrained Generator ...')
                torch.save({'state_dict': generator.state_dict(),
                            'epochs': epoch + 1,
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                            'datetime': datetime.now().isoformat()}, PT_GEN_MODEL_FILE)

    # Pretrain Discriminator
    if not args.force_pretrain and op.exists(PT_DSCR_MODEL_FILE):
        print("Loading Pretrained Discriminator ...")
        checkpoint = torch.load(PT_DSCR_MODEL_FILE)
        discriminator.load_state_dict(checkpoint['state_dict'])
        print('::INFO:: DateTime - %s.' % checkpoint['datetime'])
        print('::INFO:: Model was trained on %d data generations.' % checkpoint['data_gens'])
        print('::INFO:: Model was trained for %d epochs per data generation.' % checkpoint['epochs'])
        print('::INFO:: Final Loss - %.5f' % checkpoint['loss'])
    else:
        print('Pretraining Discriminator ...')
        dscr_criterion = nn.NLLLoss(size_average=False)
        dscr_optimizer = optim.Adam(discriminator.parameters(), lr=args.dscr_learning_rate)
        if args.cuda and torch.cuda.is_available():
            dscr_criterion = dscr_criterion.cuda()
        for i in range(DSCR_PRETRAIN_DATA_GENS):
            print('Creating real and fake datafiles ...')
            create_generated_data_file(generator, BATCH_SIZE*len(data_loader), BATCH_SIZE, GEN_FILE)
            create_real_data_file(data_loader, REAL_FILE)
            dscr_data_iter = DataLoader(DscrDataset(REAL_FILE, GEN_FILE), batch_size=BATCH_SIZE, shuffle=True)
            for j in range(DSCR_PRETRAIN_EPOCHS):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer, args.train_type)
                print('::PRETRAIN DSCR:: Data Gen [%d] Epoch [%d] Loss: %f' % (i, j, loss))
        print('Caching Pretrained Discriminator ...')
        torch.save({'state_dict': discriminator.state_dict(),
                    'data_gens': DSCR_PRETRAIN_DATA_GENS,
                    'epochs': DSCR_PRETRAIN_EPOCHS,
                    'loss': loss,
                    'datetime': datetime.now().isoformat()}, PT_DSCR_MODEL_FILE)

    # create real data file if it doesn't yet exist
    if not op.exists(REAL_FILE):
        print('Creating real data file...')
        create_real_data_file(data_loader, REAL_FILE)

    # create generated data file if it doesn't yet exist
    if not op.exists(GEN_FILE):
        print('Creating generated data file...')
        create_generated_data_file(generator, BATCH_SIZE * len(data_loader), BATCH_SIZE, GEN_FILE)

    # Adversarial Training 
    print('#'*100)
    print('Start Adversarial Training...\n')
    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss(use_cuda=args.cuda)
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_gan_optm = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
    if args.cuda and torch.cuda.is_available():
        gen_gan_loss = gen_gan_loss.cuda()
        gen_criterion = gen_criterion.cuda()

    dscr_criterion = nn.NLLLoss(size_average=False)
    dscr_optimizer = optim.Adam(discriminator.parameters(), lr=args.dscr_learning_rate)
    if args.cuda and torch.cuda.is_available():
        dscr_criterion = dscr_criterion.cuda()

    for epoch in range(GAN_TRAIN_EPOCHS):
        print("#"*30 + "\nAdversarial Epoch [%d]\n" % epoch + "#"*30)
        for gstep in range(G_STEPS):
            ## Train the generator G_STEPS
            samples = generator.sample(BATCH_SIZE, gen_seq_len)

            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if args.cuda and torch.cuda.is_available():
                zeros = zeros.cuda()
            # import pdb
            # pdb.set_trace()
            inputs = Variable(torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous())
            targets = Variable(samples.data)

            # calculate the reward
            rewards = rollout.get_reward(samples, NUM_ROLLOUTS, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            # rewards = torch.exp(rewards).contiguous().view((-1,))
            if args.cuda and torch.cuda.is_available():
                rewards = rewards.cuda()

            prob = generator.forward(inputs)
            train_loss = gen_gan_loss(prob, targets, rewards)
            print('Adv Epoch [%d], Gen Step [%d] - Train Loss: %f' % (epoch, gstep, train_loss))
            gen_gan_optm.zero_grad()
            train_loss.backward()
            gen_gan_optm.step()

            valid_loss = eval_epoch(generator, valid_loader, gen_criterion, args.train_type)
            print('Adv Epoch [%d], Gen Step [%d] - Valid Loss: %f' % (epoch, gstep, valid_loss))
            rollout.update_params()
        
        for data_gen in range(D_DATA_GENS):
            create_generated_data_file(generator, GENERATED_NUM, BATCH_SIZE, GEN_FILE)
            dscr_data_iter = DataLoader(DscrDataset(REAL_FILE, GEN_FILE), batch_size=BATCH_SIZE, shuffle=True)
            for dstep in range(D_STEPS):
                loss = train_epoch(discriminator, dscr_data_iter, dscr_criterion, dscr_optimizer, args.train_type)
                print('Adv Epoch [%d], Dscr Gen [%d], Dscr Step [%d] - Loss: %f' % (epoch, data_gen, dstep, loss))

    run_dir = op.join("runs", datetime.now().strftime('%b%d-%y_%H:%M:%S'))
    if not op.exists(run_dir):
        os.makedirs(run_dir)

    model_inputs = {'vocab_size': VOCAB_SIZE,
                    'embed_dim': gen_embed_dim,
                    'hidden_dim': gen_hidden_dim,
                    'use_cuda': False}
    json.dump(model_inputs, open(op.join(run_dir, 'model_inputs.json'), 'w'), indent=4)
    torch.save(generator.state_dict(), op.join(run_dir, 'generator_state.pt'))

if __name__ == '__main__':
    main()
