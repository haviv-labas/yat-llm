"""
Creates a character-level language-modelling dataset from a stream of text.
You shouldn't need to make any changes to this file.
"""

import torch
from torch.utils.data import Dataset
import re


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class WordDataset(Dataset):
    def __init__(self, config, data):
        # tokenize the text data into words
        self.tokens = re.findall(r"\b\w+\b", data.lower())

        # add '<UNK>' token to the set of words
        words = sorted(set(self.tokens + ["<UNK>"]))

        data_size, vocab_size = len(self.tokens), len(words)
        print("data has %d words, %d unique." % (data_size, vocab_size))

        self.stoi = {w: i for i, w in enumerate(words)}
        self.itos = {i: w for i, w in enumerate(words)}
        self.vocab_size = vocab_size
        self.config = config

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.tokens) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) words from the tokens
        chunk = self.tokens[idx : idx + self.config.block_size + 1]
        # encode every word to an integer
        dix = [self.stoi[w] for w in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
