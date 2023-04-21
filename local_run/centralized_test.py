"""Usage: `python3 local_run/centralized.py` i.e. run from the `runtime` directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.append('.')  # For making `src/` visible
import time

import numpy as np
from loguru import logger
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, f1_score, confusion_matrix

import src.solution_centralized as solution_centralized
import src.test_utils as test_utils


def env_setup():
  logger.info('Running env setup...')
  # Check that we have want we need
  logger.info('Validating that all required functions exist in solution_centralized...')
  assert hasattr(solution_centralized, 'fit') and callable(solution_centralized.fit)
  assert hasattr(solution_centralized, 'predict') and callable(solution_centralized.predict)
  # If we get here, other files should be here too


def predict(data_paths, model_dir, preds_format_path, preds_dest_path):
  logger.info('Running `solution_centralized.predict` function')
  solution_centralized.predict(**data_paths,
                               model_dir=model_dir,
                               preds_format_path=preds_format_path,
                               preds_dest_path=preds_dest_path)


def evaluate(preds_path, target_path):
  logger.info('Evaluating predictions...')
  preds_df = pd.read_csv(preds_path, index_col='pid').sort_index()
  labels_df = pd.read_csv(target_path, index_col='pid').sort_index()
  assert all(preds_df.index.values == labels_df.index.values), f'{preds_path=} & {target_path} mismatch!'
  probs = preds_df.values.reshape(-1)
  labels = labels_df.values.reshape(-1)
  preds = np.round(probs)
  # Merics
  auprc = average_precision_score(labels, probs)
  acc = np.mean(preds == labels)
  recall = recall_score(labels, preds, zero_division=0)
  f1 = f1_score(labels, preds, zero_division=0)
  _, fp, fn, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
  logger.info(f'Final {auprc=} {recall=} {acc=} {f1=} {fp=} {fn=} {len(labels)=}')


def get_data_paths(dir_path: Path):
  with open(dir_path / 'data.json') as f:
    data_json = json.load(f)
  data_paths = {k: dir_path / v for k, v in data_json.items()}
  return data_paths


def main(root_dir):
  root_dir = Path(root_dir)
  data_path = Path('data/pandemic/centralized')
  # Add log file for this run
  logger.add(root_dir / 'central_test.log')
  env_setup()

  logger.info(f'Reading data paths for pandemic/centralized ...')
  test_data_paths = get_data_paths(data_path / 'test')

  model_dir = root_dir / 'centralized_model_dir'
  model_dir.mkdir(parents=True, exist_ok=True)
  logger.info(f'Created {model_dir}...')

  ### Make predictions
  preds_format_path = data_path / 'test' / 'predictions_format.csv'
  preds_dest_path = root_dir / 'centralized_preds.csv'
  predict(test_data_paths, model_dir, preds_format_path, preds_dest_path)

  ### Evaluation (local testing only): we will then compare the predictions to the
  # ground truth labels.
  targete_path = data_path / 'test' / 'va_disease_outcome_target.csv.gz'
  evaluate(preds_dest_path, targete_path)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  # Since we are doing evaluations we have to have an input dir
  parser.add_argument('--root_dir', type=str, required=True)
  args = parser.parse_args()

  root_dir = Path(args.root_dir)
  assert root_dir.exists(), f'{root_dir} does not exist!'

  logger.info(f'Using {root_dir=} ')
  main(root_dir)
