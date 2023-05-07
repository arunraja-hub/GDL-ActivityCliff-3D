import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import MessagePassing, GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.loader import DataLoader as GeometricDataLoader
# from torch_scatter import scatter_add
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import functional as F
# from torchdrug import layers, core



from optuna.trial import TrialState
from .scoring import regression_scores, binary_classification_scores



def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    """
    Returns a tuple of integers specifying the architecture of an MLP. For example (200, 100, 100, 100, 1) specifies an MLP with input dim = 200, three hidden layers with 100 neurons each, and output dim = 1.
    """
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch

class GSNLayer(MessagePassing):
    """
    Message passing layer of the GSN from `Improving Graph Neural Network Expressivity
    via Subgraph Isomorphism Counting`_.

    This implements the GSN-v (vertex-count) variant in the original paper.

    .. _Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting:
        https://arxiv.org/pdf/2006.09252.pdf

    Parameters:
        input_dim (int): input dimension
        edge_input_dim (int): dimension of edge features
        max_cycle (int, optional): maximum size of graph substructures
        mlp_hidden_dims (list of int, optional): hidden dims of edge network and update network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, edge_input_dim, MIN_CYCLE=3, max_cycle=8, mlp_hidden_dims=None,
                 batch_norm=False, activation='relu'):
        super(GSNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.edge_input_dim = edge_input_dim
        self.node_counts_dim = max_cycle - MIN_CYCLE + 1
        if mlp_hidden_dims is None:
            mlp_hidden_dims = []

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.msg_mlp = layers.MLP(2 * input_dim + 2 * self.node_counts_dim + edge_input_dim,
                                  list(mlp_hidden_dims) + [input_dim], activation)
        self.update_mlp = layers.MLP(2 * input_dim, list(mlp_hidden_dims) + [input_dim], activation)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        node_out = graph.edge_list[:, 1]
        if graph.num_edge:
            message = torch.cat([input[node_in], input[node_out],
                                 graph.node_structural_feature[node_in].float(),
                                 graph.node_structural_feature[node_out].float(),
                                 graph.edge_feature.float()], dim=-1)
            message = self.msg_mlp(message)
        else:
            message = torch.zeros(0, self.input_dim, device=graph.device)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = torch.cat([input, update], dim=-1)
        output = self.update_mlp(output)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

class GSN(nn.Module):
    """
    Graph Substructure Network proposed in `Improving Graph Neural Network Expressivity
    via Subgraph Isomorphism Counting`_.

    This implements the GSN-v (vertex-count) variant in the original paper.

    .. _Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting:
        https://arxiv.org/pdf/2006.09252.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        edge_input_dim (int): dimension of edge features
        num_relation (int): number of relations
        num_layer (int): number of hidden layers
        num_mlp_layer (int, optional): number of MLP layers in each message passing layer
        max_cycle (int, optional): maximum size of graph substructures
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dim, edge_input_dim, num_relation, num_layer, num_mlp_layer=2, max_cycle=8,
                 short_cut=False, batch_norm=False, activation='relu', concat_hidden=False, pooling_operation=global_add_pool):
        super(GSN, self).__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feature_dim = hidden_dim * num_layer
        else:
            feature_dim = hidden_dim
        self.output_dim = feature_dim
        self.num_relation = num_relation
        self.num_layer = num_layer
        self.max_cycle = max_cycle
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.linear = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(GSNLayer(hidden_dim, edge_input_dim, max_cycle, [hidden_dim] * (num_mlp_layer - 1),
                                        batch_norm, activation))
        
        # define final pooling operation to reduce graph to vector
        self.pool = pooling_operation

        
    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        # if not hasattr(graph, 'node_structural_feature'):
        #     generate_node_structural_feature(graph, self.max_cycle)

        hiddens = []
        layer_input = self.linear(input)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return graph_feature
    # {
    #         'graph_feature': graph_feature,
    #         'node_feature': node_feature
    #     }
    



def all_combs_list(l_1, l_2):
    """
    Creates a list of all possible pairs (a,b) whereby a is in l_1 and b is in l_2.
    """
    
    all_combs = []
    
    for a in l_1:
        for b in l_2:
            all_combs.append((a,b))
   
    return all_combs


class MLP(nn.Module):
    """
    MLP class with variable architecture, implemented in PyTorch. Optionally includes batchnorm and dropout.
    """
    
    def __init__(self, 
                 architecture = (1, 10, 10, 1), 
                 hidden_activation = nn.ReLU(), 
                 output_activation = nn.Identity(), 
                 use_bias = True, 
                 hidden_dropout_rate = 0.0, 
                 hidden_batchnorm = False):
        
        # inherit initialisation method from parent class
        super(MLP, self).__init__()
        
        # define computational layers
        self.layers = nn.ModuleList()
        
        for k in range(len(architecture)-1):
            
            # add batchnorm layer
            if k > 0 and hidden_batchnorm == True:
                self.layers.append(nn.BatchNorm1d(architecture[k]))
            
            # add dropout layer
            if k > 0:
                self.layers.append(nn.Dropout(p = hidden_dropout_rate))
           
            # add affine-linear transformation layer
            self.layers.append(nn.Linear(architecture[k], architecture[k+1], bias = use_bias))
            
            # add nonlinear activation layer
            if k < len(architecture) - 2:
                self.layers.append(hidden_activation)
            else:
                self.layers.append(output_activation)
                
    def forward(self, x):
        
        # apply computational layers in forward pass
        for layer in self.layers:
            x = layer(x)
        
        return x



def fit_pytorch_mlp_model(model,
                          dataset_train,
                          dataset_test = None,
                          batch_size = 2**7,
                          dataloader_shuffle = True, 
                          dataloader_drop_last = True,
                          learning_rate = 1e-3,
                          lr_lambda = lambda epoch: 1,
                          lr_last_epoch = 0,
                          weight_decay = 1e-2,
                          num_epochs = 1,
                          loss_function = nn.MSELoss(), 
                          optimiser = torch.optim.AdamW, 
                          performance_metrics = "regression", 
                          print_results_per_epochs = 1, 
                          optuna_trial = None, 
                          optuna_performance_metric = None, 
                          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch MLP model implemented in the MLP class above. Optionally includes weight decay and learning rate decay.
    """

    # assign model to computational device
    model = model.to(device)
    
    # create dataloaders
    dataloader_train = DataLoader(dataset = dataset_train,
                                  batch_size = batch_size,
                                  shuffle = dataloader_shuffle, 
                                  drop_last = dataloader_drop_last)
    
    dataloader_train_for_eval = DataLoader(dataset = dataset_train,
                                           batch_size = len(dataset_train),
                                           shuffle = False,
                                           drop_last = False)
    
    if dataset_test != None:
        
        dataloader_test = DataLoader(dataset = dataset_test,
                                     batch_size = len(dataset_test),
                                     shuffle = False,
                                     drop_last = False)
    
    # compile optimiser and learning rate scheduler
    compiled_optimiser = optimiser(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(compiled_optimiser, lr_lambda = lr_lambda)
    
    # set learning rate scheduler state via dummy loop (in case we have trained before and want to resume training at a later epoch)
    for _ in range(lr_last_epoch):
        lr_scheduler.step()
        
    # preallocate arrays to save loss curves on training- and test set
    loss_curve_training_set = np.zeros(num_epochs)
    loss_curve_test_set = np.zeros(num_epochs)
    
    # define number of trained epochs
    trained_epochs = num_epochs
    
    # loop over training epochs
    for epoch in range(num_epochs):
        
        # set model to training mode
        model.train()
        
        # loop over minibatches for training
        for (feature_vector_batch, label_vector_batch) in dataloader_train:
            
            # assign data batch to computational device
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)

            # compute current value of loss function via forward pass
            loss_function_value = loss_function(model(feature_vector_batch), label_vector_batch)

            # set past gradient to zero
            compiled_optimiser.zero_grad()
            
            # compute current gradient via backward pass
            loss_function_value.backward()
            
            # update model weights using gradient and optimisation method
            compiled_optimiser.step()
        
        # apply learning rate scheduler
        lr_scheduler.step()
        
        # set model to evaluation mode
        model.eval()
        
        # generate current predictions and loss function values of model on training- and test set
        for (feature_vector_batch, label_vector_batch) in dataloader_train_for_eval:
            
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)
            
            y_train_pred = model(feature_vector_batch).cpu().detach().numpy()[:,0]
            y_train_true = label_vector_batch.detach().cpu().numpy()[:,0]
        
        training_loss = loss_function(torch.tensor(y_train_true, dtype = torch.float32), torch.tensor(y_train_pred, dtype = torch.float32))
        loss_curve_training_set[epoch] = training_loss
        
        if dataset_test != None:
            
            for (feature_vector_batch, label_vector_batch) in dataloader_test:

                feature_vector_batch = feature_vector_batch.to(device)
                label_vector_batch = label_vector_batch.to(device)
                
                y_test_pred = model(feature_vector_batch).cpu().detach().numpy()[:,0]
                y_test_true = label_vector_batch.detach().cpu().numpy()[:,0]

            test_loss = loss_function(torch.tensor(y_test_true, dtype = torch.float32), torch.tensor(y_test_pred, dtype = torch.float32))
            loss_curve_test_set[epoch] = test_loss
        
        # print current performance metrics (if wanted)
        if print_results_per_epochs != None:
            if epoch % print_results_per_epochs == 0:
                
                if performance_metrics == "regression":
                    print("Results after epoch ", epoch, "on training set:")
                    regression_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after epoch ", epoch, "on test set:")
                        regression_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                elif performance_metrics == "classification":
                    print("Results after epoch ", epoch, "on training set:")
                    binary_classification_scores(y_train_true, y_train_pred, display_results = True)

                    if dataset_test != None:
                        print("Results after epoch ", epoch, "on test set:")
                        binary_classification_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                else:
                    print("Neither regression- nor classification task.")

        # report intermediate results for pruning to optuna if we are inside an optuna trial
        if optuna_trial != None and optuna_performance_metric != None and dataset_test != None:

            # report intermediate result to optuna trial
            optuna_trial.report(optuna_performance_metric(y_test_true, y_test_pred), epoch)

            # decide whether trial should be pruned based on its early performance on the test set
            if optuna_trial.should_prune() == True:
                print("Pruned after epoch ", epoch, ". \n")
                raise optuna.exceptions.TrialPruned()
        
    # set model to evaluation mode
    model.eval()
            
    return (model, trained_epochs, loss_curve_training_set, loss_curve_test_set)



def train_mlps_via_optuna(dataset,
                          optuna_options,
                          mlp_hyperparameter_grid, 
                          train_hyperparameter_grid,
                          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch MLP model implemented in the MLP class above. This training loop includes an inner hyperparameter optimisation loop implemented in Optuna. 
    
    Example inputs:
    
    dataset = TensorDataset(torch.tensor(X_fp_train, dtype = torch.float), torch.tensor(Y_train, dtype = torch.float))
    
    optuna_options = {"h_iters": 20,
                  "frac_train": 0.8,
                  "data_splitting_seed": 42,
                  "performance_metric": mean_absolute_error,
                  "direction": "minimize",
                  "sampler": optuna.samplers.TPESampler(), 
                  "pruner": optuna.pruners.NopPruner()} 

    mlp_hyperparameter_grid = {"architecture": [arch(settings_dict["bitstring_length"], 1, w, d) for (w,d) in all_combs_list([64, 128, 256, 512], [1, 5, 10])],
                           "hidden_activation": [nn.ReLU()],
                           "output_activation": [nn.Identity()],
                           "use_bias": [True],
                           "hidden_dropout_rate": [0, 0.25],
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
    train_size = int(optuna_options["frac_train"] * len(dataset))
    val_size = len(dataset) - train_size
    (dataset_train, dataset_val) = torch.utils.data.random_split(dataset, 
                                                                 [train_size, val_size], 
                                                                 generator = torch.Generator().manual_seed(optuna_options["data_splitting_seed"]))
    
    # define model construction function
    def define_model(trial):
    
        model = MLP(architecture = trial.suggest_categorical("architecture", mlp_hyperparameter_grid["architecture"]), 
                    hidden_activation = trial.suggest_categorical("hidden_activation", mlp_hyperparameter_grid["hidden_activation"]), 
                    output_activation = trial.suggest_categorical("output_activation", mlp_hyperparameter_grid["output_activation"]), 
                    use_bias = trial.suggest_categorical("use_bias", mlp_hyperparameter_grid["use_bias"]), 
                    hidden_dropout_rate = trial.suggest_categorical("hidden_dropout_rate", mlp_hyperparameter_grid["hidden_dropout_rate"]), 
                    hidden_batchnorm = trial.suggest_categorical("hidden_batchnorm", mlp_hyperparameter_grid["hidden_batchnorm"]))
        
        return model

    # define objective function to be optimised
    def objective(trial):
        
        # construct the model
        model = define_model(trial)

        # train model on the training set
        (trained_model, 
         trained_epochs, 
         loss_curve_training_set, 
         loss_curve_test_set) = fit_pytorch_mlp_model(model = model,
                                                      dataset_train = dataset_train,
                                                      dataset_test = dataset_val,
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
        dataloader_val = DataLoader(dataset = dataset_val, batch_size = len(dataset_val), shuffle = False, drop_last = False)
        trained_model.eval()
        for (feature_vector_batch, label_vector_batch) in dataloader_val:
            
            feature_vector_batch = feature_vector_batch.to(device)
            label_vector_batch = label_vector_batch.to(device)
            
            y_val_pred = trained_model(feature_vector_batch).cpu().detach().numpy()[:,0]
            y_val_true = label_vector_batch.cpu().detach().numpy()[:,0]

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
    best_model = MLP(architecture = best_trial.params["architecture"],
                     hidden_activation = best_trial.params["hidden_activation"],
                     output_activation = best_trial.params["output_activation"],
                     use_bias = best_trial.params["use_bias"],
                     hidden_dropout_rate = best_trial.params["hidden_dropout_rate"],
                     hidden_batchnorm = best_trial.params["hidden_batchnorm"])
    
    # train model with best hyperparameters on whole dataset
    (trained_best_model,
     trained_epochs,
     loss_curve_training_set,
     loss_curve_test_set) = fit_pytorch_mlp_model(model = best_model,
                                                  dataset_train = dataset,
                                                  dataset_test = None,
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
    
    return (trained_best_model, loss_curve_training_set)


            
def fit_pytorch_gnn_mlp_model(gnn_model,
                              mlp_model,
                              data_list_train,
                              data_list_test = None,
                              batch_size = 2**7,
                              dataloader_shuffle = True, 
                              dataloader_drop_last = True,
                              learning_rate = 1e-3,
                              lr_lambda = lambda epoch: 1,
                              lr_last_epoch = 0,
                              weight_decay = 1e-2,
                              num_epochs = 1,
                              loss_function = nn.MSELoss(), 
                              optimiser = torch.optim.AdamW, 
                              performance_metrics = "regression", 
                              print_results_per_epochs = 1,
                              optuna_trial = None,
                              optuna_performance_metric = None, 
                              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch Geometric GIN model implemented in the GIN class above. Optionally includes weight decay and learning rate decay.
    """

    # assign models to computational device
    gnn_model = gnn_model.to(device)
    mlp_model = mlp_model.to(device)
    
    # create dataloaders for training and testing
    dataloader_train = GeometricDataLoader(dataset = data_list_train,
                                           batch_size = batch_size,
                                           shuffle = dataloader_shuffle,
                                           drop_last = dataloader_drop_last)
    
    dataloader_train_for_eval = GeometricDataLoader(dataset = data_list_train,
                                                    batch_size = len(data_list_train),
                                                    shuffle = False,
                                                    drop_last = False)
    
    if data_list_test != None:
        
        dataloader_test = GeometricDataLoader(dataset = data_list_test,
                                              batch_size = len(data_list_test),
                                              shuffle = False,
                                              drop_last = False)
    
    # compile optimiser
    compiled_optimiser = optimiser(list(gnn_model.parameters()) + list(mlp_model.parameters()), lr = learning_rate, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(compiled_optimiser, lr_lambda = lr_lambda)
    
    # set learning rate scheduler state via dummy loop (in case we have trained before and want to resume training at a later epoch)
    for _ in range(lr_last_epoch):
        lr_scheduler.step()
    
    # preallocate arrays to save loss curves on training- and test set
    loss_curve_training_set = np.zeros(num_epochs)
    loss_curve_test_set = np.zeros(num_epochs)

    # loop over training epochs
    for epoch in range(num_epochs):
        
        # set models to training mode
        gnn_model.train()
        mlp_model.train()

        # loop over minibatches for training
        for (k, batch) in enumerate(dataloader_train):
            
            # assign data batch to computational device
            batch = batch.to(device)
            
            # compute current value of loss function via forward pass
            output = mlp_model(gnn_model(batch))
            loss_function_value = loss_function(output[:,0], torch.tensor(batch.y, dtype = torch.float32))
            
            # set past gradient to zero
            compiled_optimiser.zero_grad()
            
            # compute current gradient via backward pass
            loss_function_value.backward()
            
            # update model weights using gradient and optimisation method
            compiled_optimiser.step()
        
        # apply learning rate scheduler
        lr_scheduler.step()
            
        # set models to evaluation mode
        gnn_model.eval()
        mlp_model.eval()
        
        # generate current predictions and loss function values of model on training- and test set
        for batch in dataloader_train_for_eval:
            batch = batch.to(device)
            y_train_pred = mlp_model(gnn_model(batch)).cpu().detach().numpy()[:,0]
            y_train_true = batch.y.cpu().detach().numpy()
        
        training_loss = loss_function(torch.tensor(y_train_true, dtype = torch.float32), torch.tensor(y_train_pred, dtype = torch.float32))
        loss_curve_training_set[epoch] = training_loss
        
        if data_list_test != None:
            for batch in dataloader_test:
                batch = batch.to(device)
                y_test_pred = mlp_model(gnn_model(batch)).cpu().detach().numpy()[:,0]
                y_test_true = batch.y.cpu().detach().numpy()

            test_loss = loss_function(torch.tensor(y_test_true, dtype = torch.float32), torch.tensor(y_test_pred, dtype = torch.float32))
            loss_curve_test_set[epoch] = test_loss
            
        # print current performance metrics (if wanted)
        if print_results_per_epochs != None:
            if epoch % print_results_per_epochs == 0:
                
                if performance_metrics == "regression":
                    print("Results after epoch", epoch, "on training set:")
                    regression_scores(y_train_true, y_train_pred, display_results = True)

                    if data_list_test != None:
                        print("Results after epoch", epoch, "on test set:")
                        regression_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                elif performance_metrics == "classification":
                    print("Results after epoch", epoch, "on training set:")
                    binary_classification_scores(y_train_true, y_train_pred, display_results = True)

                    if data_list_test != None:
                        print("Results after epoch", epoch, "on test set:")
                        binary_classification_scores(y_test_true, y_test_pred, display_results = True)

                    print("\n \n")

                else:
                    print("Neither regression- nor classification task.")
                    
        # report intermediate results for pruning to optuna if we are inside an optuna trial
        if optuna_trial != None and optuna_performance_metric != None and data_list_test != None:

            # report intermediate result to optuna trial
            optuna_trial.report(optuna_performance_metric(y_test_true, y_test_pred), epoch)

            # decide whether trial should be pruned based on its early performance on the test set
            if optuna_trial.should_prune() == True:
                print("Pruned after epoch ", epoch, ". \n")
                raise optuna.exceptions.TrialPruned()
                
    # set models to evaluation mode
    gnn_model.eval()
    mlp_model.eval()
        
    return (gnn_model, mlp_model, loss_curve_training_set, loss_curve_test_set)



def train_gsn_mlps_via_optuna(data_list,
                              optuna_options,
                              gin_hyperparameter_grid, 
                              mlp_hyperparameter_grid, 
                              train_hyperparameter_grid, 
                              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Training loop for PyTorch Geometric GIN model implemented in the GIN class above. This training loop includes an inner hyperparameter optimisation loop implemented in Optuna. 
    
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

    gin_hyperparameter_grid = {"n_conv_layers": [1, 2, 3],
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
        
        chosen_hidden_dim = trial.suggest_categorical("hidden_dim", gin_hyperparameter_grid["hidden_dim"])
        len_arch = len(trial.suggest_categorical("architecture", mlp_hyperparameter_grid["architecture"]))
        chosen_architecture = tuple([chosen_hidden_dim for _ in range(len_arch - 1)]) + (1,)
        
        gnn_model = GSN(n_conv_layers = trial.suggest_categorical("n_conv_layers", gin_hyperparameter_grid["n_conv_layers"]),
                        input_dim = trial.suggest_categorical("input_dim", gin_hyperparameter_grid["input_dim"]),
                        hidden_dim = chosen_hidden_dim,
                        mlp_n_hidden_layers = trial.suggest_categorical("mlp_n_hidden_layers", gin_hyperparameter_grid["mlp_n_hidden_layers"]),
                        mlp_hidden_activation = trial.suggest_categorical("mlp_hidden_activation", gin_hyperparameter_grid["mlp_hidden_activation"]),
                        mlp_output_activation = trial.suggest_categorical("mlp_output_activation", gin_hyperparameter_grid["mlp_output_activation"]),
                        mlp_use_bias = trial.suggest_categorical("mlp_use_bias", gin_hyperparameter_grid["mlp_use_bias"]),
                        mlp_hidden_dropout_rate = trial.suggest_categorical("mlp_hidden_dropout_rate", gin_hyperparameter_grid["mlp_hidden_dropout_rate"]),
                        mlp_hidden_batchnorm = trial.suggest_categorical("mlp_hidden_batchnorm", gin_hyperparameter_grid["mlp_hidden_batchnorm"]),
                        eps = trial.suggest_categorical("eps", gin_hyperparameter_grid["eps"]),
                        train_eps = trial.suggest_categorical("train_eps", gin_hyperparameter_grid["train_eps"]),
                        pooling_operation = trial.suggest_categorical("pooling_operation", gin_hyperparameter_grid["pooling_operation"]))
        
        mlp_model = MLP(architecture = chosen_architecture, 
                        hidden_activation = trial.suggest_categorical("hidden_activation", mlp_hyperparameter_grid["hidden_activation"]), 
                        output_activation = trial.suggest_categorical("output_activation", mlp_hyperparameter_grid["output_activation"]), 
                        use_bias = trial.suggest_categorical("use_bias", mlp_hyperparameter_grid["use_bias"]), 
                        hidden_dropout_rate = trial.suggest_categorical("hidden_dropout_rate", mlp_hyperparameter_grid["hidden_dropout_rate"]), 
                        hidden_batchnorm = trial.suggest_categorical("hidden_batchnorm", mlp_hyperparameter_grid["hidden_batchnorm"]))
        
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
    best_gnn_model = GSN(n_conv_layers = best_trial.params["n_conv_layers"],
                         input_dim = best_trial.params["input_dim"],
                         hidden_dim = best_trial.params["hidden_dim"],
                         mlp_n_hidden_layers = best_trial.params["mlp_n_hidden_layers"],
                         mlp_hidden_activation = best_trial.params["mlp_hidden_activation"],
                         mlp_output_activation = best_trial.params["mlp_output_activation"],
                         mlp_use_bias = best_trial.params["mlp_use_bias"],
                         mlp_hidden_dropout_rate = best_trial.params["mlp_hidden_dropout_rate"],
                         mlp_hidden_batchnorm = best_trial.params["mlp_hidden_batchnorm"],
                         eps = best_trial.params["eps"],
                         train_eps = best_trial.params["train_eps"],
                         pooling_operation = best_trial.params["pooling_operation"])
    
    best_mlp_model = MLP(architecture = tuple([best_trial.params["hidden_dim"] for _ in range(len(best_trial.params["architecture"]) - 1)]) + (1,),
                         hidden_activation = best_trial.params["hidden_activation"],
                         output_activation = best_trial.params["output_activation"],
                         use_bias = best_trial.params["use_bias"],
                         hidden_dropout_rate = best_trial.params["hidden_dropout_rate"],
                         hidden_batchnorm = best_trial.params["hidden_batchnorm"])

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
    
    
