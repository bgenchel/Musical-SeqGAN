import os.path as op
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import pdb
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from data.parsing.datasets import NottinghamDataset
from evaluation.bleu import BleuScore
from generator import Generator

VOCAB_SIZE = 89
EMBED_DIM = 64
HIDDEN_DIM = 128
SEQ_LEN = 32
BATCH_SIZE = 128

def listify(tensor):
    return tensor.cpu().numpy().tolist()

def main():
    print("Loading Data ... ")
    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', seq_len=SEQ_LEN, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=128, drop_last=True, shuffle=True)
    pretrained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=True)
    pretrained.cuda()
    pretrained.load_state_dict(torch.load(op.join('pretrained', 'generator.pt'), map_location='cpu')['state_dict'])
    fully_trained = Generator(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, use_cuda=True)
    fully_trained.load_state_dict(torch.load(op.join('runs', 'Nov27-18_14:16:33', 'generator_state.pt'), map_location='cpu'))
    fully_trained.cuda()

    pt_preds = []
    ft_preds = []
    targets = []
    print("Generating Predictions ... ")
    for (data, target) in tqdm(dataloader):
        pdb.set_trace()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        pt_pred = pretrained.forward(data).argmax(2)
        ft_pred = fully_trained.forward(data).argmax(2)
        pt_preds.extend(listify(pt_pred))
        ft_preds.extend(listify(ft_pred))
        targets.extend(listify(target))

    print("Calculating BLEU Scores")
    bs = BleuScore(SEQ_LEN)
    # pdb.set_trace()
    pt_bleu = bs.evaluate_bleu_score(pt_preds, targets)
    ft_bleu = bs.evaluate_bleu_score(ft_preds, targets)

    print("BLEU Score for pretrained generator: {}".format(pt_bleu))
    print("BLEU Score for fully_trained generator: {}".format(ft_bleu))


if __name__ == '__main__':
    main()
