import sys
from torch.utils.data import DataLoader

sys.path.append('../..')
from parsing.datasets import NottinghamDataset
from evaluation.bleu import BleuScore



SEQ_LEN = 32

def main():
    dataset = NottinghamDataset('../../../data/raw/nottingham-midi', 
                                seq_len=gen_seq_len, train_type=args.train_type, data_format="nums")
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True)
    



if __name__ == '__main__':
    main()
