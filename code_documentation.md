# Code Documentation

This file serves to document the code of team puffle's submission to the US/UK PETs Prize Challenge Track B: Pandemic Forecasting, following the guidelines outlined [here](https://www.drivendata.org/competitions/103/submissions/extra/92/).

## Directory Structure

The main submission files are in the folder `submission_src/`, which contains all the code for the centralized and federated solutions. The remainder of this repo are primarily

In side `submission_src/`:

* `solution_centralized.py`: Contains the code for running the centralized solution in the flower framework
* `solution_federated.py`: Contains the code for running the federated solution in the flower framework
* `loss_utils.py`: Contains the loss functions used for training models
* `graph_utils.py`: Contains utility functions for processing the contact network, leveraging the `graph-tool` library
* `data_utils.py`: Contains the main data preprocessing and loading functions, where `create_data_loader` is the main entry point for creating data loaders.
* `training_utils.py`: Contains the utility functions for model training and evaluation, as well as hyperparameters for the solution
* `install.sh`: Contains the startup script for our submission
* `accounting.py`: Contains the code for the differential privacy accounting

The implementation for graph processing ( `graph_utils.py` ) and data loading ( `data_utils.py` ) is shared between the centralized and federated solutions.


## Overview of Learning Algorithm and Corresponding Code Locations

On a high level, our solution is a logistic regression over neighborhood (subgraph) vectors defined for each person; these feature vectors combine the features of the person and its neighboring nodes and edges. The modeling technique may be viewed a (over-)simplified variant of a graph neural network (GNN) with a single step of non-parameterized neighborhood aggregation, followed by a single step of feature transform and prediction using a single linear layer. In the following, a "neighborhood" is synonymous with a "subgraph" of the contact network, which is a subgraph of the contact network induced by a person and its immediate 1-hop neighbors.

**High-level Steps for Training and Inference :**
(applies to both the centralized solution, and each client in the federated solution)

1. [`graph_utils.py`] For a given contact graph (specified by `population_network_data_path`), we load it into a `graph_tool.Graph` object, and collect edge attributes accordingly.
  + See `construct_graph_with_edgefeats()`

2. [`graph_utils.py`] To build a neighorhood feature vector of the same dimension for each person, we need to subsample the same number of neighbors from each person's neighborhood. For privacy and efficiency reasons, this sampling is *deterministic*, which is equivalent to sparsifying the graph and bounding its maximum degree.
  + See `sample_neighborhood_with_edgefeats()`

3. [`data_utils.py` / `graph_utils.py`] With a sampled neighborhood (a list of neighbor `pid`s and corresponding edge attributes), we can now construct a neighborhood feature vector for each person by reading relevant node features from CSVs.
  + For node features, see `graph_utils.centrality_measures`,       `data_utils.create_node_features`
  + For edge features, see `data_utils.create_graph_features`
  + For the concatenation of node and edge features into a neighborhood vector, see `data_utils.create_neighborhood_features`

4. [`data_utils.py`] To generate labels, we select a *random window* of 21 days from the time series of each person's infection status, and set the binary label as whether there is a *transition into infection* in the next 7 days as the label. For example, for every person with pid `a`, we crop her infection history from a random day `t_a` to `t_a + 21` days, and set the label as whether she is infected within `t_a + 21` to `t_a + 21 + 7` days. We crop the *same* history for all of `a`'s neighbors. The infection histories are used as node features. Note that the labels of `a`'s neighbors are *not* used as a feature.
  + For the window selection, see `data_utils.create_neighborhood_features`
  + For the label generation, see `data_utils.create_training_example`
  + For inference, we select the *last* window (i.e. day 36-57) for feature construction

5. [`data_utils.py`] We can then create a data loader for training or inference. In both cases, the dataset is the list of `pid`s within the dataset (either of a client or of the central training), and after shuffling, each `pid` is mapped to its neighborhood vector, and then mapped to a training example (with label) or an inference example (without label).
  + See `data_utils.create_data_loader`

6. [`training_utils.py` / `loss_utils.py`] We can then train a model using the data loader. The model is a single linear layer with sigmoid activation for binary classification. Since there is a heavy data imbalance, we make use of Focal Loss as a mitigation. Our main privacy mechanism is **node-level differential privacy**, which can be implemented using standard DP-SGD operating on *neighborhood feature vectors* as inputs (see our technical report for more details). With a model-agnostic dataloader, the training implementation can be done in most frameworks; we make use of JAX for efficient private training.
  + See `training_utils.train_model` for model training implementation'
  + See `loss_utils.focal_loss` for focal loss implementation

7. [`training_utils.py`] During inference, we similarly create a dataloader (without random windows over infection status) and pass the trained model and loader to `training_utils.inference_model` to generate predictions for each `pid`. The predictions are then saved to file following the runtime guideline.

Note that the above steps apply to both our centralized and federated solutions, as the overall input and output formats are the same (taking input CSV files and outputting predictions). The only difference is that the above steps are repeated for each client in the federated setting.

**Federated solution**

Our federated solution makes use of **model personalization** and **federated mean-regularized multi-task learning** (see our technical report for more details). In each round of training, the clients `fit` their personalized models locally, same as the centralized solution, but the clients *regularize their models towards the mean of the all the client models* (the mean is initially zero). Then, the server simply averages the client models and broadcasts it to all clients in the next round. Importantly, clients keep their own local models across rounds and use them (instead of the server model) for inference.

Code pointers:
  + `solution_federated.TrainClient.fit`: Trains a local model just like the central solution, and saves it to its own directory. Also includes the mean-regularization step.
  + `solution_federated.TestClient.evaluate`: Uses the local model for inference just like the central solution
  + `solution_federated.TrainStrategy.aggregate_fit`: Performs the server-side averaging of the client models.

Please see our technical report for more details on our algorithm.

## Overview of Execution Flow

**Centralized Solution**

1. [`solution_centralized.fit`]
  + Load hyperparameters from `training_utils.py`
  + Create a data loader for training using `data_utils.create_data_loader`
  + Train a model using `training_utils.train_model`
  + Save the trained model to file

2. [`solution_centralized.predict`]
  + Load the trained model from file
  + Create a data loader for inference using `data_utils.create_data_loader`
  + Inference using `training_utils.inference_model`
  + Save the predictions to file

**Federated Solution**

1. [`solution_federated.TrainClient.fit`]
  + Load hyperparameters from `training_utils.py`
  + For each client:
    - Create a data loader for training using `data_utils.create_data_loader`
    - Train a model using `training_utils.train_model`
    - Save the trained model to file

2. [`solution_federated.TestClient.evaluate`]
  + For each client:
    - Load the trained model from its own `client_dir`
    - Create a data loader for inference using `data_utils.create_data_loader`
    - Inference using `training_utils.inference_model`
    - Save the predictions to file

## `install.sh`

The setup script only involves disabling the GPU as our solution runs equally fast on CPUs.
