import argparse
import json
import os
import random
from functools import partial
from pathlib import Path
from pprint import pprint

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from jaxtyping import Float
from nnsight import LanguageModel
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

import wandb
from e2e_sae import SAETransformer
from openai_sae import sparse_autoencoder

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import warnings

warnings.filterwarnings("ignore")


# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.",
)
warnings.filterwarnings(
    "ignore",
    message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.",
)

import logging

from transformers import logging as transformers_logging

# Suppress logging messages from the `transformers` library
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
