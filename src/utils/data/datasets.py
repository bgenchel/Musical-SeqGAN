import copy
import numpy as np
import os
import os.path as op
import pickle
import pretty_midi as pm
import sys
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
import constants as const

DEFAULT_DATA_PATH = op.join(Path(op.abspath(__file__)).parents[3], 'data', 'processed', 'bebop-pkl')

# MIDI Range Stuff
Ab0 = 20 # Ab0 is rest, A0 is first possible
A0 = 21 # gonna swith over to adding the zero explicitly
C8 = 108

NOTE_TICK_LENGTH = 89

class BebopTicksDataset(Dataset):
    """
    This defines the loading and organizing of parsed MusicXML data into a 'ticks' format, meaning one token represents
    a constant division of time.
    """
    def __init__(self, load_dir, measures_per_seq=4, hop_size=4, range_low=A0, range_high=C8, data_format="nums", 
                 target_type="full_sequence", **kwargs):
        """
        Loads the MIDI tick information, groups into sequences based on measures.
        :param load_dir: location of parsed MusicXML data
        :param measures_per_seq: how many measures of ticks should be considered a sequence
        :param hop_size: how many measures to move in time when creating sequences
        :param target_type: if "full_sequence" the target is the full input sequence, if "next_step" the target is
                the last tick of the input sequence
        :param format: either numbers (midi numbers) if "nums" or one hot vectors if "vecs"
        """
        assert data_format in ("nums", "vecs")
        assert target_type in ("full_sequence", "next_step")

        super().__init__()
        self.measures_per_seq = measures_per_seq
        self.hop_size = hop_size

        if not op.exists(load_dir):
            raise Exception("Data directory does not exist.")

        self.sequences = self._create_data_dict()
        self.targets = self._create_data_dict()

        for fname in os.listdir(load_dir):
            if op.splitext(fname)[1] != ".pkl":
                print("Skipping %s..." % fname)
                continue

            song = pickle.load(open(op.join(load_dir, fname), "rb"))

            # if song["metadata"]["ticks_per_measure"] != 96:
                # print("Skipping %s because it isn't in 4/4." % fname)
            if song["metadata"]["time_signature"] != "4/4":
                print("Skipping %s because it isn't in 4/4." % fname)

            full_sequence = self._create_data_dict()
            for i, measure in enumerate(song["measures"]):
                for j, group in enumerate(measure["groups"]):
                    chord_vec = group['harmony']['root'] + group['harmony']['pitch_classes']
                    harmony = [copy.deepcopy(chord_vec) for _ in range(len(group['ticks']))]
                    full_sequence[const.CHORD_KEY].extend(harmony)

                    formatted_ticks = []
                    for tick in group['ticks']:
                        # make room for rest at the bottom of the range, 
                        # include the highest range index
                        formatted = tick[range_low - 1:range_high + 1] 
                        formatted[0] = tick[0] # translate rest into lowest position of new range
                        formatted_ticks.append(formatted)

                    if data_format == "nums":
                            full_sequence[const.TICK_KEY].extend(list(np.array(formatted_ticks).argmax(axis=0)))
                    elif data_format == "vecs":
                        full_sequence[const.TICK_KEY].extend(formatted_ticks)
            full_sequence = {k: np.array(v) for k, v in full_sequence.items()} 
            for k, seq in full_sequence.items():
                seqs, targets = self._get_seqs_and_targets(seq)
                self.sequences[k].extend(seqs)
                self.targets[k].extend(seqs)

def _get_seqs_and_targets(self, sequence):
    seqs, targets = [], []
    seq_len = self.measures_per_seq * const.TICKS_PER_MEASURE
    if len(sequence.shape) == 1:
        padding = np.zeros((seq_len))
    else:
        padding = np.zeros((seq_len, sequence.shape[1]))
    sequence = np.concatenate((padding, sequence), axis=0)
    # sequence = np.concatenate((padding, sequence), axis=1)
    for i in range(sequence.shape[0], seq_len):
        seqs.append(sequence[i:()])
        if self.target_type == 'next_step':
            targets.append(sequence[i + self.seq_len])
        elif self.target_type == 'full_sequence':
            targets.append(sequence[(i + 1):(i + self.seq_len + 1)])
    return seqs, targets

def __len__(self):
    """
    The length of the dataset.
    :return: the number of sequences in the dataset
    """
    return len(self.sequences['pitch_numbers'])

def __getitem__(self, index):
    """
    A sequence and its target.
    :param index: the index of the sequence and target to fetch
    :return: the sequence and target at the specified index
    """ 
    seqs = {k: torch.LongTensor(seqs[index]) for k, seqs in self.sequences.items()}
    seqs[const.CHORD_KEY] = seqs[const.CHORD_KEY].float()
    targets = {k: torch.LongTensor(np.array(targs[index])) for k, targs in self.targets.items()}
    targets[const.CHORD_KEY] = targets[const.CHORD_KEY].float()
    return (seqs, targets)

@staticmethod
def _create_data_dict():
    return {const.TICK_KEY: [], const.CHORD_KEY: []}

@staticmethod
def _get_none_harmony(self):
    """
    Gets a representation of no harmony.
    :return: a CHORD_ROOT_DIM + CHORD_PITCH_CLASSES_DIM x 1 list of zeroes
    """
    return [0 for _ in range(const.CHORD_ROOT_DIM + const.CHORD_PITCH_CLASSES_DIM)]

@staticmethod
def _get_empty_ticks(self, num_ticks):
    """
    Gets a representation of a rest with no harmony for a given number of ticks.
    :param num_ticks: how many ticks of rest with no harmony desired
    :return: a list of num_ticks x 61
    """
    ticks = []

    for _ in range(num_ticks):
        tick = [0 for _ in range(const.CHORD_ROOT_DIM + const.CHORD_PITCH_CLASSES_DIM + NOTE_TICK_LENGTH)]
        tick[-1] = 1
        ticks.append(tick)

    return ticks


class NottinghamDataset(Dataset):
    """
    Loads the nottingham dataset, in midi format. From the original paper:
    
    For music composition, we use the Nottingham dataset as our training data, 
    which is a collection of 695 music of folk tunes in midi fname format. We 
    study the solo track of each music. In our work, we use 88 numbers to 
    represent 88 pitches, which correspond to the 88 keys on the piano. With 
    the pitch sampling for every 0.4s, we transform the midi fnames into 
    sequences of numbers from 1 to 88 with the length 32.
    """

    def __init__(self, load_dir, period=0.4, seq_len=32, data_format="nums", target_type="full_sequence", **kwargs):
        """
        Loads the MIDI tick information, converts into format based on original paper designation.
        :param load_dir: location of nottingham midi dataset
        :param period: how much time 1 tick represents in the MIDI
        :param seq_len: how many ticks in a sequence/data point
        :param target_type: if "full_sequence", the target and sequence are both the full input sequence, 
            if "next_step", target will be the next step and sequence will be the proceeding *seq_len* steps.
        :param data_format: if "nums", then the data will be a list of note numbers from 0 to 88. if "ticks", then the 
            data will be a sequence of one-hot size 88 vectors.
        """
        super().__init__(**kwargs)

        assert data_format in ("ticks", "nums")
        assert target_type in ("full_sequence", "next_step")
        if not op.exists(load_dir):
            raise Exception("Data directory does not exist.")

        self.seq_len = seq_len
        self.data_format = data_format
        self.target_type = target_type

        self.seqs = []
        self.targets = []

        for fname in os.listdir(load_dir):
            if op.splitext(fname)[1] != ".mid":
                # print("Skipping %s..." % fname)
                continue
            song = pm.PrettyMIDI(op.join(load_dir, fname))
            melody = song.instruments[0]
            piano_roll = melody.get_piano_roll(fs=(1/period))
            piano_roll = piano_roll[Ab0:C8+1] # paper uses 88 keys, 
            self._sequence_load(piano_roll)

    def _sequence_load(self, piano_roll):
        # pad for an even split
        padding = np.zeros((piano_roll.shape[0], self.seq_len))
        piano_roll = np.concatenate((padding, piano_roll), axis=1)
        if self.data_format == "nums":
            piano_roll = np.argmax(piano_roll, axis=0) # gets rid of an axis

        for i in range(piano_roll.shape[0] - self.seq_len):
            self.seqs.append(piano_roll[i:(i + self.seq_len)])
            if self.target_type == "full_sequence": 
                self.targets.append(piano_roll[(i + 1):(i + self.seq_len + 1)])
            elif self.target_type == "next_step":
                self.targets.append(piano_roll[i + self.seq_len])

    def __len__(self):
        """
        The length of the dataset.
        :return: the number of sequences in the dataset
        """
        return len(self.seqs)

    def __getitem__(self, index):
        """
        A sequence and its target.
        :param index: the index of the sequence and target to fetch
        :return: the sequence and target at the specified index
        """
        return torch.from_numpy(np.array(self.seqs[index])), torch.from_numpy(np.array(self.targets[index]))
