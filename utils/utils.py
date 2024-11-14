
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from argparse import ArgumentParser


# Definining the rotation orthogonal layer
class RotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, init_orth=True):
        super().__init__()
        weight = torch.empty(n, n)
        if init_orth:
            torch.nn.init.orthogonal_(weight)
        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


activations = []

def forward_hook(module, input, output):
    activations.append(output[0])

# model.transformer.h[4].attn.register_forward_hook(forward_hook)

class data:
    def __init__(self, config):
        self.config = config
        self.path_continent = self.config["continent_data_path"]
        self.path_country = self.config["country_data_path"]

    def load_data(self):
        with open(self.path_continent, "r") as file:
            self.continent_data = json.load(file)

        with open(self.path_country, "r") as file:
            self.country_data = json.load(file)
        
        return self.continent_data, self.country_data

class eval:
    
    def __init__(
        self,
        model,
        tokenizer,
        config):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def eval_accuracy(self, data):
        with torch.no_grad():
            for sample in data:
                inputs = self.tokenizer(sample[0], return_tensors="pt").to(self.config["device"])
                output = self.model(inputs['input_ids'])
                output_logits = output[0]
                if output_logits.argmax(dim=-1) == sample[1]:
                    correct += 1
            return correct/len(data)
              