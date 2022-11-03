import torch

from utils import *
from models import *

import sys
import argparse
import random

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def main():
    SEED = 1234
    model_name = "multi30k"
    data_name = "multi30k"
    model_type = "seq2seqV2"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, valid_data, train_iterator, valid_iterator, SRC, TRG = dataloaders(device, data_name, batch_size = 128, seed = SEED)

    model = model_train(model_name, model_type, device, SRC, TRG, train_iterator, valid_iterator, n_epochs = 1 )
if __name__ == "__main__":
    main()