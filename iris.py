import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from utils import load_data_from_csv

csv_path = "data/Iris.csv"

X_train, X_test = load_data_from_csv(csv_path)
