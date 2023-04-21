import os
# Just use CPU for now
os.environ['CPU_OR_GPU'] = 'cpu'  # Submission specific envvar
import gc

from pathlib import Path
from pprint import pformat
from typing import Optional, Tuple, Union, List

from loguru import logger
import numpy as np
import pandas as pd

import flwr as fl
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

import src.data_utils as data_utils
import src.training_utils as training_utils

# Just set hparams as globals
FED_HPARAMS = training_utils.FED_HPARAMS
BATCH_SIZE = FED_HPARAMS['batch_size']
N_EPOCHS = FED_HPARAMS['num_epochs']
N_ROUNDS = FED_HPARAMS['num_rounds']
N_NABES = FED_HPARAMS['neighborhood_sizes']
LR = FED_HPARAMS['lr']
SEED = FED_HPARAMS['seed']
MRMTL_LAM = FED_HPARAMS['mrmtl_lam']
EARLY_STOP = FED_HPARAMS['early_stop']
assert len(N_NABES) == 1, 'Only support 1-hop neighbors for now'

# Dimensions of node, edge, history, and overall features
D_NODE = data_utils.D_NODE
D_EDGE = data_utils.D_EDGE
D_HISTORY = data_utils.D_HISTORY
D_INPUT = D_NODE * (1 + N_NABES[0]) - D_HISTORY + 1 + D_EDGE * N_NABES[0]


def train_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
  """
  Factory function that instantiates and returns a Flower Client for training.
  The federated learning simulation engine will use this function to
  instantiate clients with all necessary dependencies.

  Args:
      cid (str): Identifier for a client node/federation unit. Will be
          constant over the simulation and between train and test stages.
      person_data_path (Path): Path to CSV data file for the Person table, for
          the partition specific to this client.
      household_data_path (Path): Path to CSV data file for the House table,
          for the partition specific to this client.
      residence_location_data_path (Path): Path to CSV data file for the
          Residence Locations table, for the partition specific to this
          client.
      activity_location_data_path (Path): Path to CSV data file for the
          Activity Locations on table, for the partition specific to this
          client.
      activity_location_assignment_data_path (Path): Path to CSV data file
          for the Activity Location Assignments table, for the partition
          specific to this client.
      population_network_data_path (Path): Path to CSV data file for the
          Population Network table, for the partition specific to this client.
      disease_outcome_data_path (Path): Path to CSV data file for the Disease
          Outcome table, for the partition specific to this client.
      client_dir (Path): Path to a directory specific to this client that is
          available over the simulation. Clients can use this directory for
          saving and reloading client state.

  Returns:
      (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
  """
  logger.info(f'[{cid=}] Hyperparams: {pformat(FED_HPARAMS)}')
  return TrainClient(cid, client_dir, person_data_path, household_data_path,
                     residence_location_data_path, activity_location_data_path,
                     activity_location_assignment_data_path, population_network_data_path,
                     disease_outcome_data_path)


def train_strategy_factory(server_dir: Path) -> Tuple[fl.server.strategy.Strategy, int]:
  """
  Factory function that instantiates and returns a Flower Strategy, plus the
  number of federated learning rounds to run.

  Args:
      server_dir (Path): Path to a directory specific to the server/aggregator
          that is available over the simulation. The server can use this
          directory for saving and reloading server state. Using this
          directory is required for the trained model to be persisted between
          training and test stages.

  Returns:
      (Strategy): Instance of Flower Strategy.
      (int): Number of federated learning rounds to execute.
  """
  return TrainStrategy(server_dir), N_ROUNDS


def test_client_factory(
    cid: str,
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
  """
  Factory function that instantiates and returns a Flower Client for test-time
  inference. The federated learning simulation engine will use this function
  to instantiate clients with all necessary dependencies.

  Args:
      cid (str): Identifier for a client node/federation unit. Will be
          constant over the simulation and between train and test stages.
      person_data_path (Path): Path to CSV data file for the Person table, for
          the partition specific to this client.
      household_data_path (Path): Path to CSV data file for the House table,
          for the partition specific to this client.
      residence_location_data_path (Path): Path to CSV data file for the
          Residence Locations table, for the partition specific to this
          client.
      activity_location_data_path (Path): Path to CSV data file for the
          Activity Locations on table, for the partition specific to this
          client.
      activity_location_assignment_data_path (Path): Path to CSV data file
          for the Activity Location Assignments table, for the partition
          specific to this client.
      population_network_data_path (Path): Path to CSV data file for the
          Population Network table, for the partition specific to this client.
      disease_outcome_data_path (Path): Path to CSV data file for the Disease
          Outcome table, for the partition specific to this client.
      client_dir (Path): Path to a directory specific to this client that is
          available over the simulation. Clients can use this directory for
          saving and reloading client state.
      preds_format_path (Path): Path to CSV file matching the format you must
          write your predictions with, filled with dummy values.
      preds_dest_path (Path): Destination path that you must write your test
          predictions to as a CSV file.

  Returns:
      (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
  """
  return TestClient(cid, client_dir, preds_format_path, preds_dest_path, person_data_path,
                    household_data_path, residence_location_data_path, activity_location_data_path,
                    activity_location_assignment_data_path, population_network_data_path,
                    disease_outcome_data_path)


def test_strategy_factory(server_dir: Path, ) -> Tuple[fl.server.strategy.Strategy, int]:
  """
  Factory function that instantiates and returns a Flower Strategy, plus the
  number of federation rounds to run.

  Args:
      server_dir (Path): Path to a directory specific to the server/aggregator
          that is available over the simulation. The server can use this
          directory for saving and reloading server state. Using this
          directory is required for the trained model to be persisted between
          training and test stages.

  Returns:
      (Strategy): Instance of Flower Strategy.
      (int): Number of federated learning rounds to execute.
  """
  num_rounds = 1  # Only need to evaluate 1 round
  return TestStrategy(server_dir), num_rounds


# def flower_parameter_to_hk_params_helper(parameters: List[np.ndarray]):
#   return training_utils.

################################################################################
######################## Train Flower Classes ##################################
################################################################################


class TrainClient(fl.client.NumPyClient):
  def __init__(
      self,
      cid: str,
      client_dir: Path,
      *data_paths: List[Path],
  ):
    # Keep the initialization simple
    super().__init__()
    self.cid = cid
    self.client_dir = client_dir
    self.data_paths = data_paths
    logger.info(f'[cid={cid} Train] Received paths at init: {data_paths=}')
    self.model_fname = f'client_{cid}_params.npy'

  def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
    ### Extract server configs
    round_idx = config['round_idx']
    mrmtl_lam = config['mrmtl_lam'] if 'mrmtl_lam' in config else MRMTL_LAM
    cid = self.cid
    import jax
    logger.info(f'[{cid=}/r={round_idx} Train] Hardware check: {jax.default_backend()=} ...')

    ### First load the stateful model, if there's any, or initialize fresh params
    model = training_utils.create_model(model_fn=training_utils.model_fn)
    if (self.client_dir / self.model_fname).exists():
      logger.info(f'[{cid=}/r={round_idx} Train] Loading params from {self.client_dir=} ...')
      params = training_utils.load_params(self.client_dir, fname=self.model_fname)
    else:
      logger.info(f'[{cid=}/r={round_idx} Train] Initializing params ...')
      batch_x_template = np.zeros((BATCH_SIZE, D_INPUT))
      params = training_utils.init_model(model, batch_x_template, seed=SEED)

    params_shapes = training_utils.array_shapes(params)
    logger.info(f'[{cid=}/r={round_idx} Train] params shapes = {params_shapes}')

    ### Load server params: the only element is the model params
    # We will maintain an invariant that the server only maintains the
    # flattened params vector, and the client will maintain the structured params
    # Specifically, the server's `Parameters` == serialized(list of a single flat vector)
    server_params = training_utils.struct_unflatten(parameters[0], struct_template=params)

    ### DP params
    if 'l2_clip' in FED_HPARAMS and 'noise_mult' in FED_HPARAMS:
      l2_clip = FED_HPARAMS['l2_clip']
      noise_mult = FED_HPARAMS['noise_mult']
      logger.info(f'[{cid=}/r={round_idx} Train] ***** Using DP with '
                  f'l2_clip={l2_clip} and noise_mult={noise_mult}')
    else:
      logger.info(f'[{cid=}/r={round_idx} Train] ***** NOT Using DP')
      l2_clip, noise_mult = None, None

    ### Create dataloader
    logger.info(f'[{cid=}/r={round_idx} Train] Creating data loader v2 (with edge features)...')
    train_loader, num_examples = data_utils.create_loader(*self.data_paths,
                                                          batch_size=BATCH_SIZE,
                                                          is_prediction=False,
                                                          neighborhood_sizes=N_NABES,
                                                          client_id=cid,
                                                          return_num_examples=True,
                                                          cache_dir=self.client_dir)
    gc.collect()
    ### Train models
    logger.info(f'[{cid=}/r={round_idx} Train] Training model ...')
    client_seed = SEED + round_idx + hash(cid)
    final_params = training_utils.train_model(model,
                                              params,
                                              train_loader,
                                              N_EPOCHS,
                                              LR,
                                              mrmtl_lam=mrmtl_lam,
                                              mrmtl_params=server_params,
                                              l2_clip=l2_clip,
                                              noise_mult=noise_mult,
                                              seed=client_seed,
                                              early_stop=EARLY_STOP)
    gc.collect()
    # Finally save the model to client_dir for next round
    logger.info(f'[{cid=}/r={round_idx} Train] Saving model for {round_idx=} ...')
    training_utils.save_params(self.client_dir, final_params, fname=self.model_fname)
    # Return the trained params to the server
    # the training_utils.save/load_params and struct_flatten/unflatten should be interoperable
    flat_final_params = training_utils.struct_flatten(final_params)
    empty_metrics = {}
    logger.info(f'[{cid=}/r={round_idx} Train] {round_idx=} done!')
    return [flat_final_params], num_examples, empty_metrics


class TrainStrategy(fl.server.strategy.Strategy):
  """Federated aggregation equivalent to pooling observations across partitions.

  Maintain an invariant that the server only maintains the flattened params
  vector (i.e. Parameters = encoded([flat_params]), and the client will maintain
  the structured params.
  """
  def __init__(self, server_dir: Path):
    self.server_dir = server_dir
    super().__init__()

  def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
    logger.info(f'[Server Train] Initializing params ...')
    model = training_utils.create_model(model_fn=training_utils.model_fn)
    batch_x_template = np.zeros((BATCH_SIZE, D_INPUT))
    params = training_utils.init_model(model, batch_x_template, seed=SEED)
    # Convert to flower params
    flat_params = training_utils.struct_flatten(params)
    return fl.common.ndarrays_to_parameters([flat_params])

  def configure_fit(self, server_round: int, parameters: Parameters,
                    client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
    logger.info(f'[Server Train] Configuring fit for round {server_round}...')
    # Fit every client once. Don't need to pass any initial parameters or config.
    clients = list(client_manager.all().values())
    config_dict = {'round_idx': server_round, 'mrmtl_lam': MRMTL_LAM}
    fit_ins = fl.common.FitIns(parameters, config_dict)
    logger.info(f'[Server Train] ...done configuring fit for round {server_round}')
    return [(client, fit_ins) for client in clients]

  def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                    failures) -> Tuple[Optional[Parameters], dict]:

    if len(failures) > 0:
      raise Exception(f'Client fit round had {len(failures)} failures. Failures: {failures}')

    # results is List[Tuple[ClientProxy, FitRes]]
    # convert FitRes to List[flat_vec]
    client_flat_params = [
        fl.common.parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results
    ]
    client_num_examples = [fit_res.num_examples for _, fit_res in results]

    logger.info(f'[Server Train] Averaging client params for round {server_round}...')
    new_flat_params = np.average(client_flat_params, axis=0, weights=client_num_examples)
    new_paramaters = fl.common.ndarrays_to_parameters([new_flat_params])
    empty_metrics = {}
    gc.collect()
    return new_paramaters, empty_metrics

  def configure_evaluate(self, server_round: int, parameters, client_manager):
    """Do nothing. Return empty list."""
    # follows example_src
    return []

  def aggregate_evaluate(self, server_round: int, results, failures):
    """Do nothing. Expect no results to aggregate. Return empty results."""
    # follows example_src
    return None, {}

  def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, dict]]:
    """Write model to disk. No actual evaluation."""
    # follows example_src: just save model and no evaluation
    save_path = self.server_dir / f'server_params_r{server_round:02}.npy'
    save_path_latest = self.server_dir / 'server_params_latest.npy'
    logger.info(
        f'[Server Train] Saving model for round {server_round} to {save_path.resolve()=}...')
    flat_params = fl.common.parameters_to_ndarrays(parameters)[0]
    np.save(save_path, flat_params, allow_pickle=False)
    np.save(save_path_latest, flat_params, allow_pickle=False)
    return None


################################################################################
######################## Test Flower Classes ###################################
################################################################################


class TestClient(fl.client.NumPyClient):
  """Custom Flower NumPyClient class for test."""
  def __init__(
      self,
      cid: str,
      client_dir: Path,
      preds_format_path: Path,
      preds_dest_path: Path,
      *data_paths: List[Path],
  ):
    super().__init__()
    self.cid = cid
    self.client_dir = client_dir
    self.preds_format_path = preds_format_path
    self.preds_dest_path = preds_dest_path
    self.data_paths = data_paths
    logger.info(f'[cid={cid} Test] Received paths at init: {data_paths=}')
    self.model_fname = f'client_{cid}_params.npy'

  def evaluate(self, parameters: Parameters, config: dict) -> Tuple[float, int, dict]:
    # We do not need the server model
    round_idx = config['round_idx']
    cid = self.cid
    import jax
    logger.info(f'[{cid=}/r={round_idx} Test] Hardware check: {jax.default_backend()=} ...')

    ### Load saved *personalized* model from disk
    logger.info(f'[{cid=}/r={round_idx} Test] Loading params from {self.client_dir=} ...')
    model = training_utils.create_model(model_fn=training_utils.model_fn)
    params = training_utils.load_params(self.client_dir, fname=self.model_fname)

    ### Create inference loader
    logger.info(f'[{cid=}/r={round_idx} Test] Creating inference loader ...')
    predict_loader = data_utils.create_loader(*self.data_paths,
                                              batch_size=BATCH_SIZE,
                                              is_prediction=True,
                                              neighborhood_sizes=N_NABES,
                                              client_id=cid,
                                              cache_dir=self.client_dir)
    gc.collect()

    ### Read in predictions format and run inference
    logger.info(f'[{cid=}/r={round_idx} Test] Reading in predictions format ...')
    preds_format_df = pd.read_csv(self.preds_format_path, index_col='pid')
    logger.info(f'[{cid=}/r={round_idx} Test] Running inference ...')
    probs, pids = training_utils.inference_model(model, params, predict_loader)
    gc.collect()

    logger.info(f'[{cid=}/r={round_idx} Test] Formatting predictions ...')
    preds_format_df.loc[pids] = probs.reshape(-1, 1)  # Assignment needs extra dim

    ### Save predictions
    logger.info(f'[{cid=}/r={round_idx} Test] Saving predictions to {self.preds_dest_path=} ...')
    preds_format_df.to_csv(self.preds_dest_path)
    logger.info(f'[{cid=}/r={round_idx} Test] Evaluation done!')

    # Follow example_src: return empty metrics. We're not actually evaluating anything
    return 0.0, 0, {}


class TestStrategy(fl.server.strategy.Strategy):
  """Custom Flower strategy for test.

  Maintain an invariant that the server only maintains the flattened params
  vector (i.e. Parameters = encoded([flat_params]), and the client will maintain
  the structured params.
  """
  def __init__(self, server_dir: Path):
    self.server_dir = server_dir
    super().__init__()

  def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
    """Load saved model parameters from training."""
    save_path_latest = self.server_dir / 'server_params_latest.npy'
    if save_path_latest.exists():
      logger.info(f'[Server Test] Loading saved model from {save_path_latest=}...')
      flat_params = np.load(save_path_latest, allow_pickle=False)
    else:
      logger.info(f'[Server Test] {save_path_latest=} does not exist. Creating some dummy model...')
      model = training_utils.create_model(model_fn=training_utils.model_fn)
      batch_x_template = np.zeros((BATCH_SIZE, D_INPUT))
      params = training_utils.init_model(model, batch_x_template, seed=SEED)
      flat_params = training_utils.struct_flatten(params)

    parameters = fl.common.ndarrays_to_parameters([flat_params])
    logger.info(f'[Server Test] Loaded saved model from {save_path_latest=}!')
    return parameters

  def configure_evaluate(self, server_round: int, parameters: Parameters,
                         client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
    """Run evaluate on all clients to make test predictions."""
    logger.info(f'[Server Test] Configuring evaluate for round {server_round}...')
    config_dict = {'round_idx': server_round}
    evaluate_ins = fl.common.EvaluateIns(parameters, config_dict)
    clients = list(client_manager.all().values())
    return [(client, evaluate_ins) for client in clients]

  def configure_fit(self, server_round, parameters, client_manager):
    """Do nothing and return empty list. We don't need to fit clients for test."""
    # follows example_src
    return []

  def aggregate_fit(self, server_round, results, failures):
    """Do nothing and return empty results. No fit results to aggregate for test."""
    # follows example_src
    return None, {}

  def aggregate_evaluate(self, server_round, results, failures):
    """Do nothing and return empty results. Not actually evaluating any metrics."""
    # follows example_src
    return None, {}

  def evaluate(self, server_round, parameters):
    # follows example_src
    return None
