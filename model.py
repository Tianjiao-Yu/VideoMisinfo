import tensorflow as tf

import torch
import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm_notebook
from uuid import uuid4

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# If there's a GPU available...
if torch.cuda.is_available():  
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# Load the dataset into a pandas dataframe.
df_train = pd.read_csv('./papadamou/train_metadata.csv',encoding='UTF-8')
df_test = pd.read_csv('./papadamou/test_metadata.csv',encoding='UTF-8')



df_train = df_train.sample(frac=1).reset_index(drop=True)

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))

# Display 10 random rows from the data.
print(df_train.sample(10))
print(len(df_test))