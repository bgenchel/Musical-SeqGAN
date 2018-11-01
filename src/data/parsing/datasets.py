import numpy as np
import os
import os.path as op
import pickle
import pretty_midi as pm
import torch
from torch.utils.data.dataset import Dataset

# MIDI Range Stuff
Ab0 = 20 # would start with A0, but want 0 to be below the range of possible numbers
C8 = 108

NOTE_TICK_LENGTH = 37
HARMONY_ROOT_LENGTH = 12
HARMONY_PITCH_CLASSES_LENGTH = 12

class MidiTicksDataset(Dataset):
    """
    This defines the loading and organizing of parsed MusicXML data into a MIDI tick database.
    """

    def __init__(self, load_dir, measures_per_seq=8, hop_size=4, target_type="full_sequence", **kwargs):
        """
        Loads the MIDI tick information, groups into sequences based on measures.
        :param load_dir: location of parsed MusicXML data
        :param measures_per_seq: how many measures of ticks should be considered a sequence
        :param hop_size: how many measures to move in time when creating sequences
        :param target_type: if "full_sequence" the target is the full input sequence, if "next_step" the target is
                the last tick of the input sequence
        """
        super(MidiTicksDataset, self).__init__(**kwargs)

        if not op.exists(load_dir):
            raise Exception("Data directory does not exist.")

        self.sequences = []
        self.targets = []

        for file in os.listdir(load_dir):
            if op.splitext(file)[1] != ".pkl":
                print("Skipping %s..." % file)
                continue

            song = pickle.load(open(op.join(load_dir, file), "rb"))

            if song["metadata"]["ticks_per_measure"] != 96:
                print("Skipping %s because it isn't in 4/4." % file)

            sequence_start = 0
            while sequence_start < len(song["measures"]):
                sequence = []
                for sequence_index in range(measures_per_seq):
                    measure_index = sequence_start + sequence_index
                    if measure_index < len(song["measures"]):
                        measure = song["measures"][sequence_start + sequence_index]

                        formatted_measure = []

                        # Append harmony root and pitch classes to each tick
                        for group in measure["groups"]:
                            for tick in group["ticks"]:
                                if group["harmony"]:
                                    formatted_tick = group["harmony"]["root"] + group["harmony"]["pitch_classes"] + tick
                                else:
                                    formatted_tick = self._get_none_harmony() + tick
                                formatted_measure.append(formatted_tick)

                        # Ensure measure has enough ticks
                        if measure["num_ticks"] < 96:
                            empty_ticks = self._get_empty_ticks(96 - measure["num_ticks"])

                            # If first measure, prepend padding to the measure
                            if measure_index == 0:
                                formatted_measure = empty_ticks + formatted_measure
                            # Otherwise, it should be the last measure, so append padding to the measure
                            else:
                                formatted_measure += empty_ticks

                        sequence += formatted_measure

                    else:
                        formatted_measure = self._get_empty_ticks(96)
                        sequence += formatted_measure

                if target_type == "full_sequence":
                    target = sequence
                else: # target_type == "next_step"
                    target = sequence[-1]

                self.sequences.append(sequence)
                self.targets.append(target)

                sequence_start += hop_size

    def __len__(self):
        """
        The length of the dataset.
        :return: the number of sequences in the dataset
        """
        return len(self.sequences)

    def __getitem__(self, index):
        """
        A sequence and its target.
        :param index: the index of the sequence and target to fetch
        :return: the sequence and target at the specified index
        """
        return torch.from_numpy(np.array(self.sequences[index])), torch.from_numpy(np.array(self.targets[index]))

    def _get_none_harmony(self):
        """
        Gets a representation of no harmony.
        :return: a HARMONY_ROOT_LENGTH + HARMONY_PITCH_CLASSES_LENGTH x 1 list of zeroes
        """
        return [0 for _ in range(HARMONY_ROOT_LENGTH + HARMONY_PITCH_CLASSES_LENGTH)]

    def _get_empty_ticks(self, num_ticks):
        """
        Gets a representation of a rest with no harmony for a given number of ticks.
        :param num_ticks: how many ticks of rest with no harmony desired
        :return: a list of num_ticks x 61
        """
        ticks = []

        for _ in range(num_ticks):
            tick = [0 for _ in range(HARMONY_ROOT_LENGTH + HARMONY_PITCH_CLASSES_LENGTH + NOTE_TICK_LENGTH)]
            tick[-1] = 1
            ticks.append(tick)

        return ticks


class NottinghamDataset(Dataset):
    """
    Loads the nottingham dataset, in midi format. From the original paper:
    
    For music composition, we use the Nottingham dataset as our training data, 
    which is a collection of 695 music of folk tunes in midi file format. We 
    study the solo track of each music. In our work, we use 88 numbers to 
    represent 88 pitches, which correspond to the 88 keys on the piano. With 
    the pitch sampling for every 0.4s, we transform the midi files into 
    sequences of numbers from 1 to 88 with the length 32.
    """

    def __init__(self, load_dir, period=0.4, seq_len=32, data_format="nums", train_type="full_sequence", **kwargs):
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
        super(NottinghamDataset, self).__init__(**kwargs)

        assert data_format in ("ticks", "nums")
        assert train_type in ("full_sequence", "next_step")
        if not op.exists(load_dir):
            raise Exception("Data directory does not exist.")

        self.seq_len = seq_len
        self.data_format = data_format
        self.train_type = train_type

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
            if self.train_type == "full_sequence": 
                self.targets.append(piano_roll[(i + 1):(i + self.seq_len + 1)])
            elif self.train_type == "next_step":
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
