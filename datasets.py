import torch
from torch.utils.data import Dataset
import numpy as np
from helper import one_hot_encode, get_batches


# character dataset targeted for RNN uses
class CharacterDataset(Dataset):

    def __init__(self, text, vocabulary, batch_size=10, seq_length=50, device="cpu"):
        self.original_text = text
        batch_char_size = batch_size * seq_length
        # character length of the trimmed text
        self.text_length = (len(text) // batch_char_size) * batch_char_size

        self.batch_size = batch_size
        self.seq_length = seq_length

        # character tokenization
        self.vocabulary = vocabulary
        # character to integer translation dictionaries
        self.int2char = dict(enumerate(vocabulary))
        self.char2int = {ch: i for i, ch in self.int2char.items()}

        # character to integer translation
        encoded_text = np.array([self.char2int[ch] for ch in self.original_text])
        # we prepare Xs (inputs) and Ys (targets) beforehand to save computation time
        self.x_y = [(
            torch.tensor(one_hot_encode(batch[0], len(vocabulary))).to(device),
            torch.tensor(batch[1]).to(device)
        )
            for batch in get_batches(encoded_text, batch_size, seq_length)]

    def __getitem__(self, idx):
        return self.x_y[idx]

    def __len__(self):
        return int(self.text_length / (self.batch_size * self.seq_length))

    def change_device(self, device):
        for i, (x, y) in enumerate(self.x_y):
            x = x.to(device)
            y = y.to(device)

            self.x_y[i] = x, y
