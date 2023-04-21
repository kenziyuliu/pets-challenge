import os
# Just use CPU for now
os.environ['CPU_OR_GPU'] = 'cpu'  # Submission specific envvar
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force no GPU for central only
import gc

from pathlib import Path
from pprint import pformat
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd

import src.data_utils as data_utils
import src.training_utils as training_utils

### Constants
CENTRAL_HPARAMS = training_utils.CENTRAL_HPARAMS

EARLY_STOP = CENTRAL_HPARAMS['early_stop']
N_NABES = CENTRAL_HPARAMS['neighborhood_sizes']
D_NODE = data_utils.D_NODE
D_EDGE = data_utils.D_EDGE
D_HISTORY = data_utils.D_HISTORY
D_INPUT = D_NODE * (1 + N_NABES[0]) - D_HISTORY + 1 + D_EDGE * N_NABES[0]


def fit(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
) -> None:
  import jax
  logger.info(f'Checking hardware: {jax.default_backend()=}')

  # Add logging to model_dir
  logger.add(model_dir / 'central_fit.log')
  logger.info(f'Hyperparams: {CENTRAL_HPARAMS}')

  batch_size = CENTRAL_HPARAMS['batch_size']
  neighborhood_sizes = CENTRAL_HPARAMS['neighborhood_sizes']
  lr = CENTRAL_HPARAMS['lr']
  num_epochs = CENTRAL_HPARAMS['num_epochs']
  seed = CENTRAL_HPARAMS['seed']

  # DP params
  if 'l2_clip' in CENTRAL_HPARAMS and 'noise_mult' in CENTRAL_HPARAMS:
    l2_clip = CENTRAL_HPARAMS['l2_clip']
    noise_mult = CENTRAL_HPARAMS['noise_mult']
    logger.info(f'Using DP with l2_clip={l2_clip} and noise_mult={noise_mult}')
  else:
    l2_clip, noise_mult = None, None

  logger.info(f'Creating centralized data loader v2 (with edge features)...')
  train_loader = data_utils.create_loader(person_data_path,
                                          household_data_path,
                                          residence_location_data_path,
                                          activity_location_data_path,
                                          activity_location_assignment_data_path,
                                          population_network_data_path,
                                          disease_outcome_data_path,
                                          batch_size=batch_size,
                                          is_prediction=False,
                                          neighborhood_sizes=neighborhood_sizes,
                                          cache_dir=model_dir)
  gc.collect()

  # Create model and params
  logger.info(f'Creating model ...')
  model = training_utils.create_model(model_fn=training_utils.model_fn)
  n_nabes = 1 + sum(neighborhood_sizes)  # including self node
  batch_x_template = np.zeros((batch_size, D_INPUT))
  logger.info(f'Initializing model ...')
  params = training_utils.init_model(model, batch_x_template, seed=seed)
  params_shapes = training_utils.array_shapes(params)
  logger.info(f'params shapes = {params_shapes}')

  # Train model
  logger.info(f'Training model ...')
  final_params = training_utils.train_model(model,
                                            params,
                                            train_loader,
                                            num_epochs,
                                            lr,
                                            l2_clip=l2_clip,
                                            noise_mult=noise_mult,
                                            early_stop=EARLY_STOP)

  # Save model
  logger.info(f'Saving model ...')
  training_utils.save_params(model_dir, final_params, fname='central_params.npy')
  logger.info(f'Central training done!')


def predict(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> None:
  import jax
  logger.info(f'Checking hardware: {jax.default_backend()=}')

  # Add logging to model_dir
  logger.add(model_dir / 'central_predict.log')
  # Create model and load params
  logger.info(f'Creating model and loading params ...')
  model = training_utils.create_model(model_fn=training_utils.model_fn)
  params = training_utils.load_params(model_dir, fname='central_params.npy')

  batch_size = CENTRAL_HPARAMS['batch_size']
  neighborhood_sizes = CENTRAL_HPARAMS['neighborhood_sizes']

  logger.info(f'Creating centralized data loader ...')
  predict_loader = data_utils.create_loader(person_data_path,
                                            household_data_path,
                                            residence_location_data_path,
                                            activity_location_data_path,
                                            activity_location_assignment_data_path,
                                            population_network_data_path,
                                            disease_outcome_data_path,
                                            batch_size=batch_size,
                                            is_prediction=True,
                                            neighborhood_sizes=neighborhood_sizes,
                                            cache_dir=model_dir)

  # Read preds_format
  preds_format_df = pd.read_csv(preds_format_path, index_col='pid')
  logger.info(f'Predicting ...')
  probs, pids = training_utils.inference_model(model, params, predict_loader)
  preds_format_df.loc[pids] = probs.reshape(-1, 1)  # Assignment needs extra dim

  # Save predictions
  logger.info(f'Saving predictions ...')
  preds_format_df.to_csv(preds_dest_path)
  logger.info(f'Central predict done!')
