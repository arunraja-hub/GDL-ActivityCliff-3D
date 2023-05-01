import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import warnings
import argparse

# RDKit
from rdkit import Chem, RDLogger

# scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV

from modules import *


RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', metavar='N', type=str, nargs='+',
                    help='an integer for the accumulator')
# chembl_dopamine_d2
# chembl_factor_xa
# postera_sars_cov_2_mpro
args = parser.parse_args()

settings_dict = {}
datafolder_filepath = "/vols/opig/users/raja/GDL-ActivityCliff-3D/data/"+args.dataset[0]
settings_dict["target_name"] = args.dataset[0]

if args.dataset[0] == "postera_sars_cov_2_mpro":
    activity_type = "f_avg_IC50 [uM]"
else: 
    activity_type = "Ki [nM]"


dataframe = pd.read_csv(datafolder_filepath + "/molecule_data_clean.csv", sep = ",")
settings_dict["n_molecules"] = len(dataframe)

# construct array with SMILES strings
x_smiles = dataframe["SMILES"].values
np.save(datafolder_filepath +'/x_smiles',x_smiles)

y = -np.log10(dataframe[activity_type].values.astype(float))

# create dictionary which maps SMILES strings to their activity labels
x_smiles_to_y_dict = dict(list(zip(x_smiles, y)))

# import matched molecular pair (MMP) dataframe
dataframe_mmps = pd.read_csv(datafolder_filepath + "/MMP_data_clean.csv",
                             sep = ",",
                             header = 0)

# construct MMP array and binary MMP labels for AC-prediction (we delete half-cliffs)

# create array with all MMPs (including half-cliffs)
X_smiles_mmps = dataframe_mmps.values[:,0:2]

# create array with all MMP cores
x_smiles_mmp_cores = dataframe_mmps.values[:,3]

# label ACs with 1 and other MMPs (= half-cliffs and non-ACs) with 0
y_mmps = np.array([int(abs(x_smiles_to_y_dict[smiles_1] - x_smiles_to_y_dict[smiles_2]) >= 2) 
                   for [smiles_1, smiles_2] in X_smiles_mmps])

# determine indices for MMPs which are half-cliffs (which will be deleted- see ind_delete)
y_mmps_half_cliffs = np.array([int(1 < abs(x_smiles_to_y_dict[smiles_1] - x_smiles_to_y_dict[smiles_2]) < 2) 
                               for [smiles_1, smiles_2] in X_smiles_mmps])
ind_delete = np.ndarray.flatten(np.argwhere(y_mmps_half_cliffs > 0))

# delete MMPs which are half-cliffs
X_smiles_mmps = np.delete(X_smiles_mmps, ind_delete, axis = 0)
settings_dict["n_mmps"] = len(X_smiles_mmps)
x_smiles_mmp_cores = np.delete(x_smiles_mmp_cores, ind_delete, axis = 0)
y_mmps = np.delete(y_mmps, ind_delete)

# construct array with activity values for mmps
Y_mmps_vals = np.array([[x_smiles_to_y_dict[smiles_1], x_smiles_to_y_dict[smiles_2]] for [smiles_1, smiles_2] in X_smiles_mmps])

# randomly flip or maintain direction of all smiles pairs in X_smiles_mmps and Y_mmps_vals to make potency directionality classification balanced
np.random.seed(42)
for k in range(len(X_smiles_mmps)):
    if np.random.uniform(0,1) > 0.5:
        X_smiles_mmps[k, 0:2] = np.flip(X_smiles_mmps[k, 0:2])
        Y_mmps_vals[k, 0:2] = np.flip(Y_mmps_vals[k, 0:2])  

# construct potency directionality target variable (0: smiles_1 activity is larger than or equal to smiles_2 activity, 1: smiles_1 activity is smaller than smiles_2 activity)
y_mmps_pd = np.array([int(val_1 < val_2) for [val_1, val_2] in Y_mmps_vals])

# create data split dictionary for k-fold cross validation repeated with m random seeds
settings_dict["k_splits"] = 2
settings_dict["m_reps"] = 3
settings_dict["random_state_cv"] = 42

data_split_dictionary = create_data_split_dictionary_for_mols_and_mmps(x_smiles,
                                                                       X_smiles_mmps,
                                                                       x_smiles_mmp_cores,
                                                                       k_splits = settings_dict["k_splits"],
                                                                       m_reps = settings_dict["m_reps"],
                                                                       random_state_cv = settings_dict["random_state_cv"])
save_dict(data_split_dictionary,datafolder_filepath +'/data_split_dictionary.txt')

save_dict(settings_dict,datafolder_filepath+'/settings_dict.txt')

# saving X_smiles_mmps,y, y_mmps, y_mmps_pd
save_dict(X_smiles_mmps,datafolder_filepath +'/X_smiles_mmps.txt')
# np.savetxt('data/'+ args.dataset[0]+'/X_smiles_mmps.txt', X_smiles_mmps)
np.savetxt(datafolder_filepath +"/y.txt", y)
np.savetxt(datafolder_filepath +'/y_mmps.txt', y_mmps)
np.savetxt(datafolder_filepath +'/y_mmps_pd.txt', y_mmps_pd)
print("completed setting up", settings_dict["target_name"])
