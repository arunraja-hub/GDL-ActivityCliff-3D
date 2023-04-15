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
import argparse
import os
import os.path

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
parser.add_argument('--dataset', type=str, help='Description of arg1')
parser.add_argument('--model', type=str, help='Description of arg2')

args = parser.parse_args()
#rf
#knn
#gin

datafolder_filepath = "data/"+args.dataset

settings_dict = load_dict(datafolder_filepath + "/settings_dict.txt")
settings_dict["target_name"] = args.dataset



x_smiles = np.load(datafolder_filepath + '/x_smiles.npy', allow_pickle=True)
X_smiles_mmps = load_dict(datafolder_filepath + '/X_smiles_mmps.txt')
y = np.loadtxt(datafolder_filepath + '/y.txt')
y_mmps = np.loadtxt(datafolder_filepath + '/y_mmps.txt')
y_mmps_pd = np.loadtxt(datafolder_filepath + '/y_mmps_pd.txt')

data_split_dictionary = load_dict(datafolder_filepath + '/data_split_dictionary.txt')


# set directory for saving of experimental results

settings_dict["method_name"] = "pdv_rf"
filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"

# create PDVs
settings_dict["descriptor_list"] = None # use default 200 descriptors from literature

X_pdv = list(range(settings_dict["n_molecules"]))
for (k, smiles) in enumerate(x_smiles):
    X_pdv[k] = rdkit_mol_descriptors_from_smiles(smiles, descriptor_list = settings_dict["descriptor_list"])
X_pdv = np.array(X_pdv)

# replace NaN with 0
print("Number of NaN values to replace = ", np.sum(np.isnan(X_pdv)),"(",100*np.sum(np.isnan(X_pdv))/(X_pdv.shape[0]*X_pdv.shape[1]) ,r"%)")
X_pdv = np.nan_to_num(X_pdv)
print("Shape of X_pdv = ", X_pdv.shape)
np.save(datafolder_filepath +'/X_pdv',X_pdv)

# create dictionary that maps SMILES strings to PDVs
x_smiles_to_pdv_dict = dict(list(zip(x_smiles, X_pdv)))

if args.model == "rf":
    # set directory for saving of experimental results
    settings_dict["method_name"] = "pdv_rf"
    filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"
   
   # hyperparameter- and random search settings

    settings_dict["j_splits"] = 5
    settings_dict["h_iters"] = 10
    settings_dict["random_search_scoring"] = "neg_mean_absolute_error"
    settings_dict["random_search_verbose"] = 1
    settings_dict["random_search_random_state"] = 42
    settings_dict["random_search_n_jobs"] = -1
    settings_dict["hyperparameter_grid"] = {"n_estimators": [500], 
                                            "max_depth": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, None],
                                            "min_samples_split": [2, 4, 6, 8, 10, 12],
                                            "min_samples_leaf": [1, 2, 3, 4, 5, 6],
                                            "max_features": ["auto", "sqrt", "log2"],
                                            "bootstrap": [True, False],
                                            "random_state": [42]}
    
        # model evaluation via m-times repeated k-fold cross validation
    start_time = time.time()

    # preallocate dictionary with cubic arrays used to save prediction values and performance scores over all m*k experiments
    scores_dict = create_scores_dict(k_splits = settings_dict["k_splits"], 
                                    m_reps = settings_dict["m_reps"], 
                                    len_y = settings_dict["n_molecules"])

    # train and evaluate models
    for (m, k) in data_split_dictionary.keys():
        
        # extract indices for D_train and D_test for this data split
        (ind_train_mols, ind_test_mols) = data_split_dictionary[(m,k)][0:2]
        
        # generate training- and test data (mols)        
        X_pdv_train = X_pdv[ind_train_mols]
        y_train = y[ind_train_mols]
        
        X_pdv_test = X_pdv[ind_test_mols]
        y_test = y[ind_test_mols]
        
        # normalise data with empirical cumulative distribution functions for each feature (ecdf derived from training set)
        (X_pdv_norm_train, normalisation_function) = normaliser_cdf(X_pdv_train)
        X_pdv_norm_test = normalisation_function(X_pdv_test)
        X_pdv_norm = normalisation_function(X_pdv)
        
        # instantiate fresh model
        regressor = RandomizedSearchCV(estimator = RandomForestRegressor(),
                                    param_distributions = settings_dict["hyperparameter_grid"],
                                    n_iter = settings_dict["h_iters"],
                                    cv = settings_dict["j_splits"],
                                    scoring = settings_dict["random_search_scoring"],
                                    verbose = settings_dict["random_search_verbose"],
                                    random_state = settings_dict["random_search_random_state"],
                                    n_jobs = settings_dict["random_search_n_jobs"])

        # fit the model on the training data
        regressor.fit(X_pdv_norm_train, y_train)
        
        # create qsar predictions used to evaluate the model
        y_pred = regressor.predict(X_pdv_norm)
        create_and_store_qsar_ac_pd_results(scores_dict, x_smiles, X_smiles_mmps,
                                            y, y_mmps, y_mmps_pd, y_pred,
                                            data_split_dictionary, m, k)
        
        # give feedback on completion of this subexperiment
        print("Subexperiment ", (m,k), " completed. \n")

    # save experimental results
    save_qsar_ac_pd_results(filepath, scores_dict)

    # save experimental settings
    settings_dict["runtime"] = str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    save_experimental_settings(filepath, settings_dict)


if args.model == "knn":
   # set directory for saving of experimental results
    settings_dict["method_name"] = "pdv_knn"
    filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"
    
    # hyperparameter- and random search settings

    settings_dict["j_splits"] = 5
    settings_dict["h_iters"] = 10
    settings_dict["random_search_scoring"] = "neg_mean_absolute_error"
    settings_dict["random_search_verbose"] = 1
    settings_dict["random_search_random_state"] = 42
    settings_dict["random_search_n_jobs"] = -1
    settings_dict["hyperparameter_grid"] = {"n_neighbors": list(range(1, 101)),
                                            "weights": ["uniform", "distance"],
                                            "p": [1, 2, 3]}
        
        # model evaluation via m-times repeated k-fold cross validation
    start_time = time.time()

    # preallocate dictionary with cubic arrays used to save prediction values and performance scores over all m*k experiments
    scores_dict = create_scores_dict(k_splits = settings_dict["k_splits"], 
                                    m_reps = settings_dict["m_reps"], 
                                    len_y = settings_dict["n_molecules"])

    # train and evaluate models
    for (m, k) in data_split_dictionary.keys():
        
        # extract indices for D_train and D_test for this data split
        (ind_train_mols, ind_test_mols) = data_split_dictionary[(m,k)][0:2]
        
        # generate training- and test data (mols)        
        X_pdv_train = X_pdv[ind_train_mols]
        y_train = y[ind_train_mols]
        
        X_pdv_test = X_pdv[ind_test_mols]
        y_test = y[ind_test_mols]
        
        # normalise data with empirical cumulative distribution functions for each feature (ecdf derived from training set)
        (X_pdv_norm_train, normalisation_function) = normaliser_cdf(X_pdv_train)
        X_pdv_norm_test = normalisation_function(X_pdv_test)
        X_pdv_norm = normalisation_function(X_pdv)
        
        # instantiate fresh model
        regressor = RandomizedSearchCV(estimator = KNeighborsRegressor(),
                                    param_distributions = settings_dict["hyperparameter_grid"],
                                    n_iter = settings_dict["h_iters"],
                                    cv = settings_dict["j_splits"],
                                    scoring = settings_dict["random_search_scoring"],
                                    verbose = settings_dict["random_search_verbose"],
                                    random_state = settings_dict["random_search_random_state"],
                                    n_jobs = settings_dict["random_search_n_jobs"])

        # fit the model on the training data
        regressor.fit(X_pdv_norm_train, y_train)
        
        # create qsar predictions used to evaluate the model
        y_pred = regressor.predict(X_pdv_norm)
        create_and_store_qsar_ac_pd_results(scores_dict, x_smiles, X_smiles_mmps,
                                            y, y_mmps, y_mmps_pd, y_pred,
                                            data_split_dictionary, m, k)
        
        # give feedback on completion of this subexperiment
        print("Subexperiment ", (m,k), " completed. \n")

    # save experimental results
    save_qsar_ac_pd_results(filepath, scores_dict)

    # save experimental settings
    settings_dict["runtime"] = str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    save_experimental_settings(filepath, settings_dict)

if args.model == "mlp":
    # set directory for saving of experimental results
    settings_dict["method_name"] = "pdv_mlp"
    filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"
    
    # hyperparameter- and optuna options
    settings_dict["optuna_options"] = {"h_iters": 20,
                                    "frac_train": 0.8,
                                    "data_splitting_seed": 42,
                                    "performance_metric": mean_absolute_error,
                                    "direction": "minimize",
                                    "sampler": optuna.samplers.TPESampler(), 
                                    "pruner": optuna.pruners.NopPruner()} 

    settings_dict["mlp_hyperparameter_grid"] = {"architecture": [arch(200, 1, w, d) for (w,d) in all_combs_list([64, 128, 256, 512], [1, 5, 10])],
                                            "hidden_activation": [nn.ReLU()],
                                            "output_activation": [nn.Identity()],
                                            "use_bias": [True],
                                            "hidden_dropout_rate": [0, 0.25],
                                            "hidden_batchnorm": [True]}

    settings_dict["train_hyperparameter_grid"] = {"batch_size": [32, 64, 128],
                                                "dataloader_shuffle": [True],
                                                "dataloader_drop_last":[True],
                                                "learning_rate": [1e-2, 1e-3],
                                                "lr_lambda": [lambda epoch: max(0.95**epoch, 1e-2), lambda epoch: max(0.99**epoch, 1e-2)],
                                                "weight_decay": [0.1, 0.01],
                                                "num_epochs": [500],
                                                "loss_function": [nn.MSELoss()],
                                                "optimiser": [torch.optim.AdamW],
                                                "performance_metrics": ["regression"],
                                                "print_results_per_epochs": [None]}

    # model evaluation via m-times repeated k-fold cross validation
    start_time = time.time()

    # preallocate dictionary with cubic arrays used to save prediction values and performance scores over all m*k experiments
    scores_dict = create_scores_dict(k_splits = settings_dict["k_splits"], 
                                    m_reps = settings_dict["m_reps"], 
                                    len_y = settings_dict["n_molecules"])

    # train and evaluate models
    for (m, k) in data_split_dictionary.keys():
        
        # extract indices for D_train and D_test for this data split
        (ind_train_mols, ind_test_mols) = data_split_dictionary[(m,k)][0:2]
        
        # generate training- and test data (mols)        
        X_pdv_train = X_pdv[ind_train_mols]
        y_train = y[ind_train_mols]
        Y_train = np.reshape(y_train, (-1, 1))
        
        X_pdv_test = X_pdv[ind_test_mols]
        y_test = y[ind_test_mols]
        Y_test = np.reshape(y_test, (-1, 1))
        
        # normalise data with empirical cumulative distribution functions for each feature (ecdf derived from training set)
        (X_pdv_norm_train, normalisation_function) = normaliser_cdf(X_pdv_train)
        X_pdv_norm_test = normalisation_function(X_pdv_test)
        X_pdv_norm = normalisation_function(X_pdv)
        
        # create pytorch dataset objects for training and testing
        dataset_train = TensorDataset(torch.tensor(X_pdv_norm_train, dtype = torch.float), torch.tensor(Y_train, dtype = torch.float))
        dataset_test = TensorDataset(torch.tensor(X_pdv_norm_test, dtype = torch.float), torch.tensor(Y_test, dtype = torch.float))
        
        # find best hyperparameters via optuna and train associated model on training set
        (regressor, 
        loss_curve_training_set) = train_mlps_via_optuna(dataset_train,
                                                        settings_dict["optuna_options"],
                                                        settings_dict["mlp_hyperparameter_grid"], 
                                                        settings_dict["train_hyperparameter_grid"])
        # plot learning curves
        plt.plot(loss_curve_training_set)
        plt.title("loss curve on training set")
        plt.show()
        
        # create qsar predictions used to evaluate the model
        y_pred = regressor(torch.tensor(X_pdv_norm, dtype = torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')).cpu().detach().numpy()[:,0]
        create_and_store_qsar_ac_pd_results(scores_dict, x_smiles, X_smiles_mmps,
                                            y, y_mmps, y_mmps_pd, y_pred,
                                            data_split_dictionary, m, k)
        
        # give feedback on completion of this subexperiment
        print("Subexperiment ", (m,k), " completed. \n")

    # save experimental results
    save_qsar_ac_pd_results(filepath, scores_dict)

    # save experimental settings
    settings_dict["runtime"] = str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    save_experimental_settings(filepath, settings_dict)