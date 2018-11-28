import os.path as op
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append('../..')
from parsing.datasets import NottinghamDataset
from evaluation.bleu import BleuScore
from generator import Generator

VOCAB_SIZE = 89
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ_LEN = 32

def main():
    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', 
                                seq_len=gen_seq_len, train_type=args.train_type, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True)
    pretrained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=False)
    pretrained.load_state_dict(torch.load(op.join('pretrained', 'generator.pt')))
    pretrained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=False)
    

    



if __name__ == '__main__':
    main()
