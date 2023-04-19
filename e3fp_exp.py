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


# set E3FP hyperparameters
settings_dict["radius"] = 2
settings_dict["bitstring_length"] = 2**11
settings_dict["use_features"] = False
settings_dict["use_chirality"] = True

# set directory for saving of experimental results
# e3fp_rf
settings_dict["method_name"] = args.dataset
filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"


# create dictionary that maps SMILES strings to E3FPs
x_smiles_to_fp_dict = {}

if os.path.isfile(datafolder_filepath +'/smiles_e3fp_dict.pkl'):
    x_smiles_to_fp_dict = load_dict(datafolder_filepath +'/smiles_e3fp_dict.pkl')
    print('e3fp smiles loaded')
    for j in x_smiles_to_fp_dict.keys():
        x_smiles_to_fp_dict[j] = np.array(x_smiles_to_fp_dict[j][0].to_rdkit())
#     # breakpoint()

# if os.path.isfile('data/smiles_e3fp_dict.pkl'):
#     x_smiles_to_fp_dict = load_dict('data/smiles_e3fp_dict.pkl')
#     print('e3fp smiles loaded')
#     for j in x_smiles_to_fp_dict.keys():
#         x_smiles_to_fp_dict[j] = np.array(x_smiles_to_fp_dict[j][0].to_rdkit())
# #     breakpoint()

else:
    print('generating e3fp smiles')
    for smiles in x_smiles:
        x_smiles_to_fp_dict.update({smiles : e3fp_from_smiles(smiles)})
        with open(datafolder_filepath +'/smiles_e3fp_dict.pkl', 'wb') as f:
            pickle.dump(x_smiles_to_fp_dict,f)

if args.model == "rf":
    # set directory for saving of experimental results
    settings_dict["method_name"] = "e3fp_rf"
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
        
        # generate training- and test data (for individual molecules) 
        print(len(x_smiles))    
        print(len(x_smiles_to_fp_dict))   
        X_fp_train = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_train_mols]])
        y_train = y[ind_train_mols]
        
        X_fp_test = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_test_mols]])
        y_test = y[ind_test_mols]
        
        
        # instantiate fresh model
        regressor = RandomizedSearchCV(estimator = RandomForestRegressor(),
                                    param_distributions = settings_dict["hyperparameter_grid"],
                                    n_iter = settings_dict["h_iters"],
                                    cv = settings_dict["j_splits"],
                                    scoring = settings_dict["random_search_scoring"],
                                    verbose = settings_dict["random_search_verbose"],
                                    random_state = settings_dict["random_search_random_state"],
                                    n_jobs = settings_dict["random_search_n_jobs"])
        
        # breakpoint()
        # fit the model on the training data
        regressor.fit(X_fp_train, y_train)
        
        # create and store qsar, ac, and pd-predictions
        y_pred = regressor.predict(np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles]))
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
    settings_dict["method_name"] = "e3fp_knn"
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
        X_fp_train = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_train_mols]])
        y_train = y[ind_train_mols]
        
        X_fp_test = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_test_mols]])
        y_test = y[ind_test_mols]
        
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
        regressor.fit(X_fp_train, y_train)
        
        # create and store qsar, ac, and pd-predictions
        y_pred = regressor.predict(np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles]))
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
    settings_dict["method_name"] = "e3fp_mlp"
    filepath = "results/" + settings_dict["target_name"] + "/" + settings_dict["method_name"] + "/"

    # hyperparameter- and optuna options

    settings_dict["optuna_options"] = {"h_iters": 20,
                                    "frac_train": 0.8,
                                    "data_splitting_seed": 42,
                                    "performance_metric": mean_absolute_error,
                                    "direction": "minimize",
                                    "sampler": optuna.samplers.TPESampler(), 
                                    "pruner": optuna.pruners.NopPruner()} 

    settings_dict["mlp_hyperparameter_grid"] = {"architecture": [arch(settings_dict["bitstring_length"], 1, w, d) for (w,d) in all_combs_list([64, 128, 256, 512], [1, 5, 10])],
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
        X_fp = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles])
        
        X_fp_train = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_train_mols]])
        y_train = y[ind_train_mols]
        Y_train = np.reshape(y_train, (-1, 1))
        
        X_fp_test = np.array([x_smiles_to_fp_dict[smiles] for smiles in x_smiles[ind_test_mols]])
        y_test = y[ind_test_mols]
        Y_test = np.reshape(y_test, (-1, 1))
        
        # create pytorch dataset objects for training and testing
        dataset_train = TensorDataset(torch.tensor(X_fp_train, dtype = torch.float), torch.tensor(Y_train, dtype = torch.float))
        dataset_test = TensorDataset(torch.tensor(X_fp_test, dtype = torch.float), torch.tensor(Y_test, dtype = torch.float))
        
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
        y_pred = regressor(torch.tensor(X_fp, dtype = torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')).cpu().detach().numpy()[:,0]
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


display_experimental_results(filepath,settings_dict["target_name"]+'/'+settings_dict["method_name"], decimals = 4)
    
                





