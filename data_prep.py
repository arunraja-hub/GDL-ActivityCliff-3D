# import packages

# general tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import warnings
import json 
import torch
import torch_geometric

# RDKit
from rdkit import Chem, RDLogger

# scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

from modules import *


RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

settings_dict = {}

# ChEMBL dopamine D2 data set


datafolder_filepath = "data/chembl_dopamine_d2/"
settings_dict["target_name"] = "chembl_dopamine_d2"
activity_type = "Ki [nM]"