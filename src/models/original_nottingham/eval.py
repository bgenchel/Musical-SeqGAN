import os.path as op
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from data.parsing.datasets import NottinghamDataset
from evaluation.bleu import BleuScore
from generator import Generator

VOCAB_SIZE = 89
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ_LEN = 32

def main():
    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', seq_len=SEQ_LEN, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True)
    pretrained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=False)
    pretrained.load_state_dict(torch.load(op.join('pretrained', 'generator.pt'), map_location='cpu')['state_dict'])
    fully_trained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=False)
    fully_trained.load_state_dict(torch.load(op.join('runs', 'Nov27-18_14:16:33', 'generator_state.pt'), map_location='cpu'))

    pt_preds = []
    ft_preds = []
    targets = []
    for (data, target) in dataloader:
        pt_pred = pretrained.forward(data)
        ft_pred = fully_trained.forward(data)
        pt_preds.append(pt_pred)
        ft_preds.append(ft_pred)
        targets.append(target)

    bs = BleuScore(SEQ_LEN)
    pt_bleu = BleuScore.evaluate_bleu_score(pt_preds, targets)
    ft_bleu = BleuScore.evaluate_bleu_score(ft_preds, targets)

    print("BLEU Score for pretrained generator: {}".format(pt_bleu))
    print("BLEU Score for fully_trained generator: {}".format(ft_bleu))


if __name__ == '__main__':
    main()
