import torch
import random
import numpy as np


class DataProcessing:
    def __init__(self, data):
        data = data.replace("\n", " ")
        self.data = data
        self.num_chars = len(data)
        self.unique_chars = sorted(set(data))
        self.num_unique_chars = len(self.unique_chars)
        self.char2num = {x: i for i, x in enumerate(list(self.unique_chars))}
        self.num2char = {i: x for i, x in enumerate(list(self.unique_chars))}

    def get_random_chunk(self, chunk_len=10000):
        """
        generate a random chunk with chunk len
        """
        start_index = random.randint(0, self.num_chars - chunk_len)
        end_index = start_index + chunk_len + 1
        return self.data[start_index:end_index]

    def make_one_hot(self, text):
        """
        text is a string, subset from the original text if needed
        """
        one_hot = np.zeros((len(text), self.num_unique_chars))
        for i, char in enumerate(text):
            one_hot[i][self.char2num[char]] = 1
        return one_hot

    def get_inps_tars(self, seq_len=25):
        text = self.get_random_chunk()
        input_seq = []
        target_seq = []

        # seq creation for the chunk
        for i in range(len(text)):
            if len(text) - i > seq_len:
                input_seq.append(text[i: i + seq_len])
                target_seq.append(text[i + 1: i + seq_len + 1])

        # char to index
        for i in range(len(input_seq)):
            input_seq[i] = self.make_one_hot(input_seq[i])
            target_seq[i] = [self.char2num[char] for char in target_seq[i]]
        return torch.Tensor(input_seq), torch.Tensor(target_seq)
