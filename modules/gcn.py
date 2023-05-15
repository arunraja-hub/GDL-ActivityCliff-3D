import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader
from optuna.trial import TrialState
from .scoring import regression_scores, binary_classification_scores
from .deep_learning_pytorch import *

def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    """
    Returns a tuple of integers specifying the architecture of an MLP. For example (200, 100, 100, 100, 1) specifies an MLP with input dim = 200, three hidden layers with 100 neurons each, and output dim = 1.
    """
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch

class MLP(nn.Module):
    """
    MLP class with variable architecture, implemented in PyTorch. Optionally includes batchnorm and dropout.
    """
    
    def __init__(self, 
                 input_dim = 79,
                 ):
        
        # inherit initialisation method from parent class
        super(MLP, self).__init__()

        self.out_dim_lin = 16
        self.fc1 = nn.Linear(input_dim, self.out_dim_lin)
        self.fc2 = nn.Linear(self.out_dim_lin, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.dropout_connect = 0.5
        self.dropout_layer = nn.Dropout(self.dropout_connect)
                
    def forward(self, x):
        # add some fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout_layer(x)

        out = self.out(x)
        return x


# GCN based model
class GCN(torch.nn.Module):
    def __init__(self, trial, n_output=1, num_features_xd=78):

        super(GCN, self).__init__()

        self.trial = trial
        self.n_output = n_output
        self.out_dim_lin = 16
        # 2 ** self.trial.suggest_int("output_dim_power_linear", 4, 11)
        self.relu = F.relu

        # define the ranges for the hyper parameters
        self.number_GNN_layers = 2
        # self.trial.suggest_int("n_GNN_layers", 1, 10)
        self.act = nn.ReLU()
        # self.trial.suggest_categorical("activation_functions", ["relu", "leaky_relu"])
        self.activation = activation_function_dict[self.act]
        # we also optimize the dropout rate for the connecting layers
        

        # SMILES graph branch
        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()
        input_dim = self.num_features_xd
        for i, layer in enumerate(range(self.number_GNN_layers)):
            out_dim_power = 2
            # self.trial.suggest_int(f"output_dim_power_{layer}", 0, 2, log=False)
            self.GNN_layers.append(GCNConv(input_dim, self.num_features_xd * 2 ** out_dim_power))
            self.BN_layers.append(nn.BatchNorm1d(self.num_features_xd * 2 ** out_dim_power))
            input_dim = self.num_features_xd * 2 ** out_dim_power

        # self.fc1 = nn.Linear(input_dim, self.out_dim_lin)
        # self.fc2 = nn.Linear(self.out_dim_lin, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        # self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, (layer, bn) in enumerate(zip(self.GNN_layers, self.BN_layers)):
            x = layer(x, edge_index)
            x = self.activation(x)
            x = bn(x)

        x = global_max_pool(x, batch)       # global max pooling

        # # add some fully connected layers
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout_layer(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.dropout_layer(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.dropout_layer(x)

        # out = self.out(x)
        
        return x

    

    
# def fit_pytorch_gnn_mlp_model(gnn_model,
#                               mlp_model,
#                               data_list_train,
#                               data_list_test = None,
#                               batch_size = 2**7,
#                               dataloader_shuffle = True, 
#                               dataloader_drop_last = True,
#                               learning_rate = 1e-3,
#                               lr_lambda = lambda epoch: 1,
#                               lr_last_epoch = 0,
#                               weight_decay = 1e-2,
#                               num_epochs = 1,
#                               loss_function = nn.MSELoss(), 
#                               optimiser = torch.optim.AdamW, 
#                               performance_metrics = "regression", 
#                               print_results_per_epochs = 1,
#                               optuna_trial = None,
#                               optuna_performance_metric = None, 
#                               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     """
#     Training loop for PyTorch Geometric GCN model implemented in the GCN class above. Optionally includes weight decay and learning rate decay.
#     """

#     # assign models to computational device
#     gnn_model = gnn_model.to(device)
#     mlp_model = mlp_model.to(device)
    
#     # create dataloaders for training and testing
#     dataloader_train = GeometricDataLoader(dataset = data_list_train,
#                                            batch_size = batch_size,
#                                            shuffle = dataloader_shuffle,
#                                            drop_last = dataloader_drop_last)
    
#     dataloader_train_for_eval = GeometricDataLoader(dataset = data_list_train,
#                                                     batch_size = len(data_list_train),
#                                                     shuffle = False,
#                                                     drop_last = False)
    
#     if data_list_test != None:
        
#         dataloader_test = GeometricDataLoader(dataset = data_list_test,
#                                               batch_size = len(data_list_test),
#                                               shuffle = False,
#                                               drop_last = False)
    
#     # compile optimiser
#     compiled_optimiser = optimiser(list(gnn_model.parameters()) + list(mlp_model.parameters()), lr = learning_rate, weight_decay = weight_decay)
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(compiled_optimiser, lr_lambda = lr_lambda)
    
#     # set learning rate scheduler state via dummy loop (in case we have trained before and want to resume training at a later epoch)
#     for _ in range(lr_last_epoch):
#         lr_scheduler.step()
    
#     # preallocate arrays to save loss curves on training- and test set
#     loss_curve_training_set = np.zeros(num_epochs)
#     loss_curve_test_set = np.zeros(num_epochs)

#     # loop over training epochs
#     for epoch in range(num_epochs):
        
#         # set models to training mode
#         gnn_model.train()
#         mlp_model.train()

#         # loop over minibatches for training
#         for (k, batch) in enumerate(dataloader_train):
            
#             # assign data batch to computational device
#             batch = batch.to(device)
            
#             # compute current value of loss function via forward pass
#             output = mlp_model(gnn_model(batch))
#             loss_function_value = loss_function(output[:,0], torch.tensor(batch.y, dtype = torch.float32))
            
#             # set past gradient to zero
#             compiled_optimiser.zero_grad()
            
#             # compute current gradient via backward pass
#             loss_function_value.backward()
            
#             # update model weights using gradient and optimisation method
#             compiled_optimiser.step()
        
#         # apply learning rate scheduler
#         lr_scheduler.step()
            
#         # set models to evaluation mode
#         gnn_model.eval()
#         mlp_model.eval()
        
#         # generate current predictions and loss function values of model on training- and test set
#         for batch in dataloader_train_for_eval:
#             batch = batch.to(device)
#             y_train_pred = mlp_model(gnn_model(batch)).cpu().detach().numpy()[:,0]
#             y_train_true = batch.y.cpu().detach().numpy()
        
#         training_loss = loss_function(torch.tensor(y_train_true, dtype = torch.float32), torch.tensor(y_train_pred, dtype = torch.float32))
#         loss_curve_training_set[epoch] = training_loss
        
#         if data_list_test != None:
#             for batch in dataloader_test:
#                 batch = batch.to(device)
#                 y_test_pred = mlp_model(gnn_model(batch)).cpu().detach().numpy()[:,0]
#                 y_test_true = batch.y.cpu().detach().numpy()

#             test_loss = loss_function(torch.tensor(y_test_true, dtype = torch.float32), torch.tensor(y_test_pred, dtype = torch.float32))
#             loss_curve_test_set[epoch] = test_loss
            
#         # print current performance metrics (if wanted)
#         if print_results_per_epochs != None:
#             if epoch % print_results_per_epochs == 0:
                
#                 if performance_metrics == "regression":
#                     print("Results after epoch", epoch, "on training set:")
#                     regression_scores(y_train_true, y_train_pred, display_results = True)

#                     if data_list_test != None:
#                         print("Results after epoch", epoch, "on test set:")
#                         regression_scores(y_test_true, y_test_pred, display_results = True)

#                     print("\n \n")

#                 elif performance_metrics == "classification":
#                     print("Results after epoch", epoch, "on training set:")
#                     binary_classification_scores(y_train_true, y_train_pred, display_results = True)

#                     if data_list_test != None:
#                         print("Results after epoch", epoch, "on test set:")
#                         binary_classification_scores(y_test_true, y_test_pred, display_results = True)

#                     print("\n \n")

#                 else:
#                     print("Neither regression- nor classification task.")
                    
#         # report intermediate results for pruning to optuna if we are inside an optuna trial
#         if optuna_trial != None and optuna_performance_metric != None and data_list_test != None:

#             # report intermediate result to optuna trial
#             optuna_trial.report(optuna_performance_metric(y_test_true, y_test_pred), epoch)

#             # decide whether trial should be pruned based on its early performance on the test set
#             if optuna_trial.should_prune() == True:
#                 print("Pruned after epoch ", epoch, ". \n")
#                 raise optuna.exceptions.TrialPruned()
                
#     # set models to evaluation mode
#     gnn_model.eval()
#     mlp_model.eval()
        
#     return (gnn_model, mlp_model, loss_curve_training_set, loss_curve_test_set)



def train_gcn_mlps_via_optuna(data_list,
                              optuna_options,
                              gcn_hyperparameter_grid, 
                              mlp_hyperparameter_grid, 
                              train_hyperparameter_grid, 
                              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch Geometric GCN model implemented in the GCN class above. This training loop includes an inner hyperparameter optimisation loop implemented in Optuna. 
    
    Example inputs:
    
    graph_list = create_pytorch_geometric_data_set_from_smiles_and_targets(x_smiles, y)
    data_list = [graph_list[k] for k in ind_train_mols]
    
    optuna_options = {"h_iters": 20,
                  "frac_train": 0.8,
                  "data_splitting_seed": 42,
                  "performance_metric": mean_absolute_error,
                  "direction": "minimize",
                  "sampler": optuna.samplers.TPESampler(), 
                  "pruner": optuna.pruners.NopPruner()} 

    gcn_hyperparameter_grid = {"n_conv_layers": [1, 2, 3],
                           "input_dim": [79],
                           "hidden_dim": [64, 128, 256],
                           "mlp_n_hidden_layers": [2],
                           "mlp_hidden_activation": [nn.ReLU()],
                           "mlp_output_activation": [nn.Identity()],
                           "mlp_use_bias": [True],
                           "mlp_hidden_dropout_rate": [0, 0.25],
                           "mlp_hidden_batchnorm": [True],
                           "eps": [0],
                           "train_eps": [False],
                           "pooling_operation": [global_max_pool]}

    mlp_hyperparameter_grid = {"architecture": [arch(None, 1, w, d) for (w,d) in all_combs_list([None], [1, 5, 10])],
                           "hidden_activation": [nn.ReLU()],
                           "output_activation": [nn.Identity()],
                           "use_bias": [True],
                           "hidden_dropout_rate": [0],
                           "hidden_batchnorm": [True]}

    train_hyperparameter_grid = {"batch_size": [32, 64, 128], 
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
    """
    
    # set verbosity
    optuna.logging.set_verbosity(0)
    
    # split dataset into training and validation sets
    train_size = int(optuna_options["frac_train"] * len(data_list))
    val_size = len(data_list) - train_size
    (data_list_train, data_list_val) = torch.utils.data.random_split(data_list, [train_size, val_size],
                                                                     generator = torch.Generator().manual_seed(optuna_options["data_splitting_seed"]))
    
    # define model construction function
    def define_model(trial):
        
        # chosen_hidden_dim = trial.suggest_categorical("hidden_dim", gcn_hyperparameter_grid["hidden_dim"])
        # len_arch = len(trial.suggest_categorical("architecture", mlp_hyperparameter_grid["architecture"]))
        # chosen_architecture = tuple([chosen_hidden_dim for _ in range(len_arch - 1)]) + (1,)
        
        gnn_model = GCN(trial)
        # GCN(n_conv_layers = trial.suggest_categorical("n_conv_layers", gcn_hyperparameter_grid["n_conv_layers"]),
        #                 input_dim = trial.suggest_categorical("input_dim", gcn_hyperparameter_grid["input_dim"]),
        #                 hidden_dim = chosen_hidden_dim,
        #                 mlp_n_hidden_layers = trial.suggest_categorical("mlp_n_hidden_layers", gcn_hyperparameter_grid["mlp_n_hidden_layers"]),
        #                 mlp_hidden_activation = trial.suggest_categorical("mlp_hidden_activation", gcn_hyperparameter_grid["mlp_hidden_activation"]),
        #                 mlp_output_activation = trial.suggest_categorical("mlp_output_activation", gcn_hyperparameter_grid["mlp_output_activation"]),
        #                 mlp_use_bias = trial.suggest_categorical("mlp_use_bias", gcn_hyperparameter_grid["mlp_use_bias"]),
        #                 mlp_hidden_dropout_rate = trial.suggest_categorical("mlp_hidden_dropout_rate", gcn_hyperparameter_grid["mlp_hidden_dropout_rate"]),
        #                 mlp_hidden_batchnorm = trial.suggest_categorical("mlp_hidden_batchnorm", gcn_hyperparameter_grid["mlp_hidden_batchnorm"]),
        #                 eps = trial.suggest_categorical("eps", gcn_hyperparameter_grid["eps"]),
        #                 train_eps = trial.suggest_categorical("train_eps", gcn_hyperparameter_grid["train_eps"]),
        #                 pooling_operation = trial.suggest_categorical("pooling_operation", gcn_hyperparameter_grid["pooling_operation"]))
        
        mlp_model = MLP()
        # architecture = chosen_architecture, 
        #                 hidden_activation = trial.suggest_categorical("hidden_activation", mlp_hyperparameter_grid["hidden_activation"]), 
        #                 output_activation = trial.suggest_categorical("output_activation", mlp_hyperparameter_grid["output_activation"]), 
        #                 use_bias = trial.suggest_categorical("use_bias", mlp_hyperparameter_grid["use_bias"]), 
        #                 hidden_dropout_rate = trial.suggest_categorical("hidden_dropout_rate", mlp_hyperparameter_grid["hidden_dropout_rate"]), 
        #                 hidden_batchnorm = trial.suggest_categorical("hidden_batchnorm", mlp_hyperparameter_grid["hidden_batchnorm"]))
        
        return (gnn_model, mlp_model)

    # define objective function to be optimised
    def objective(trial):
        
        # construct the model
        (gnn_model, mlp_model) = define_model(trial)

        # train model on the training set
        (trained_gnn_model, 
         trained_mlp_model, 
         loss_curve_training_set, 
         loss_curve_test_set) = fit_pytorch_gnn_mlp_model(gnn_model = gnn_model,
                                                          mlp_model = mlp_model,
                                                          data_list_train = data_list_train,
                                                          data_list_test = data_list_val,
                                                          batch_size = trial.suggest_categorical("batch_size", train_hyperparameter_grid["batch_size"]),
                                                          dataloader_shuffle = trial.suggest_categorical("dataloader_shuffle", train_hyperparameter_grid["dataloader_shuffle"]),
                                                          dataloader_drop_last = trial.suggest_categorical("dataloader_drop_last", train_hyperparameter_grid["dataloader_drop_last"]),
                                                          learning_rate = trial.suggest_categorical("learning_rate", train_hyperparameter_grid["learning_rate"]),
                                                          lr_lambda = trial.suggest_categorical("lr_lambda", train_hyperparameter_grid["lr_lambda"]),
                                                          weight_decay = trial.suggest_categorical("weight_decay", train_hyperparameter_grid["weight_decay"]),
                                                          num_epochs = trial.suggest_categorical("num_epochs", train_hyperparameter_grid["num_epochs"]),
                                                          loss_function = trial.suggest_categorical("loss_function", train_hyperparameter_grid["loss_function"]),
                                                          optimiser = trial.suggest_categorical("optimiser", train_hyperparameter_grid["optimiser"]),
                                                          performance_metrics = trial.suggest_categorical("performance_metrics", train_hyperparameter_grid["performance_metrics"]),
                                                          print_results_per_epochs = trial.suggest_categorical("print_results_per_epochs", train_hyperparameter_grid["print_results_per_epochs"]),
                                                          optuna_trial = trial,
                                                          optuna_performance_metric = optuna_options["performance_metric"], 
                                                          device = device)
        
        # compute performance of trained model on validation set
        dataloader_val = GeometricDataLoader(dataset = data_list_val, batch_size = len(data_list_val), shuffle = False, drop_last = False)
        trained_gnn_model.eval()
        trained_mlp_model.eval()
        for batch in dataloader_val:
            batch = batch.to(device)
            y_val_pred = trained_mlp_model(trained_gnn_model(batch)).cpu().detach().numpy()[:,0]
            y_val_true = batch.y.cpu().detach().numpy()
        
        performance_measure = optuna_options["performance_metric"](y_val_true, y_val_pred)
        
        print("Trial completed. \n")
        
        return performance_measure
    
    # create hyperparameter optimisation study
    study = optuna.create_study(direction = optuna_options["direction"], 
                                sampler = optuna_options["sampler"], 
                                pruner = optuna_options["pruner"])
    
    # run hyperparameter optimisation study
    study.optimize(objective, 
                   n_trials = optuna_options["h_iters"])
    
    # print information associated with conducted study
    best_trial = study.best_trial
    print("\nStudy statistics: ")
    print("  Number of trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(study.get_trials(deepcopy = False, states = [TrialState.PRUNED])))
    print("  Number of completed trials: ", len(study.get_trials(deepcopy = False, states = [TrialState.COMPLETE])))
    print("\nHyperarameters of best trial: ")
    for (key, value) in best_trial.params.items():
        print("    {}: {}".format(key, value))
    
    # instantiate model with best hyperparameters
    best_gnn_model = GCN()
    # n_conv_layers = best_trial.params["n_conv_layers"],
    #                      input_dim = best_trial.params["input_dim"],
    #                      hidden_dim = best_trial.params["hidden_dim"],
    #                      mlp_n_hidden_layers = best_trial.params["mlp_n_hidden_layers"],
    #                      mlp_hidden_activation = best_trial.params["mlp_hidden_activation"],
    #                      mlp_output_activation = best_trial.params["mlp_output_activation"],
    #                      mlp_use_bias = best_trial.params["mlp_use_bias"],
    #                      mlp_hidden_dropout_rate = best_trial.params["mlp_hidden_dropout_rate"],
    #                      mlp_hidden_batchnorm = best_trial.params["mlp_hidden_batchnorm"],
    #                      eps = best_trial.params["eps"],
    #                      train_eps = best_trial.params["train_eps"],
    #                      pooling_operation = best_trial.params["pooling_operation"])
    
    best_mlp_model = MLP()
    # architecture = tuple([best_trial.params["hidden_dim"] for _ in range(len(best_trial.params["architecture"]) - 1)]) + (1,),
    #                      hidden_activation = best_trial.params["hidden_activation"],
    #                      output_activation = best_trial.params["output_activation"],
    #                      use_bias = best_trial.params["use_bias"],
    #                      hidden_dropout_rate = best_trial.params["hidden_dropout_rate"],
    #                      hidden_batchnorm = best_trial.params["hidden_batchnorm"])

    # train model with best hyperparameters on whole dataset
    (trained_best_gnn_model,
     trained_best_mlp_model,
     loss_curve_training_set,
     loss_curve_test_set) = fit_pytorch_gnn_mlp_model(gnn_model = best_gnn_model,
                                                      mlp_model = best_mlp_model,
                                                      data_list_train = data_list,
                                                      data_list_test = None,
                                                      batch_size = best_trial.params["batch_size"],
                                                      dataloader_shuffle = best_trial.params["dataloader_shuffle"],
                                                      dataloader_drop_last = best_trial.params["dataloader_drop_last"],
                                                      learning_rate = best_trial.params["learning_rate"],
                                                      lr_lambda = best_trial.params["lr_lambda"],
                                                      weight_decay = best_trial.params["weight_decay"],
                                                      num_epochs = best_trial.params["num_epochs"],
                                                      loss_function = best_trial.params["loss_function"],
                                                      optimiser = best_trial.params["optimiser"],
                                                      performance_metrics = best_trial.params["performance_metrics"],
                                                      print_results_per_epochs = best_trial.params["print_results_per_epochs"], 
                                                      device = device)
    
    return (trained_best_gnn_model, trained_best_mlp_model, loss_curve_training_set)
    
    
