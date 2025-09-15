########################
# Dataloader definition:
########################

import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os

class AllenCahnDataset(Dataset):
    def __init__(self, which="train", training_samples = 256):

        assert training_samples<=256

        #Default file:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_data = os.path.join(project_root, "dataset", "AllenCahn_NEW.h5")
        #self.file_data = "../dataset/AllenCahn_NEW.h5"
        self.reader = h5py.File(self.file_data, 'r')

        #Load normaliation constants:
        self.min_data = self.reader['min_u0'][()]
        self.max_data = self.reader['max_u0'][()]
        self.min_model = self.reader['min_u'][()]
        self.max_model = self.reader['max_u'][()]

        if which == "train":
            self.length = training_samples
            self.start = 0
        elif which == "val":
            self.length = 128
            self.start = 256
        elif which == "test":
            self.length = 128
            self.start = 256+128

        self.reader = h5py.File(self.file_data, 'r')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["input"][:]).type(torch.float32).reshape(1, 64, 64)
        labels = torch.from_numpy(self.reader['Sample_' + str(index + self.start)]["output"][:]).type(torch.float32).reshape(1, 64, 64)

        inputs = (inputs - self.min_data)/(self.max_data - self.min_data)
        labels = (labels - self.min_model)/(self.max_model - self.min_model)

        return inputs, labels
