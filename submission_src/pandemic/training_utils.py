import os

os.environ['CPU_OR_GPU'] = 'cpu'  # Submission specific envvar
import gc

import sys
from functools import partial
from pathlib import Path
import pickle
import time
import warnings
from typing import Optional

warnings.filterwarnings('ignore')  # Removes sklearn warnings

import chex
import haiku as hk
from loguru import logger
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad, value_and_grad, device_put
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves
import numpy as np
import optax
from sklearn.metrics import average_precision_score, recall_score, f1_score, confusion_matrix
from scipy.special import expit as np_sigmoid

import tensorflow as tf  # For dataloading only
from tqdm import tqdm

from src.loss_utils import focal_loss, forward_focal_loss

###############################################
############### Hyperparameters ###############
###############################################

FOCAL_ALPHA = 0.75
LOG_FREQ = 500

CENTRAL_HPARAMS = {}
CENTRAL_HPARAMS['log_freq'] = LOG_FREQ
CENTRAL_HPARAMS['batch_size'] = 512
CENTRAL_HPARAMS['lr'] = 0.2
CENTRAL_HPARAMS['num_epochs'] = 2
CENTRAL_HPARAMS['neighborhood_sizes'] = [20]
CENTRAL_HPARAMS['focal_alpha'] = FOCAL_ALPHA
CENTRAL_HPARAMS['seed'] = 1234
CENTRAL_HPARAMS['early_stop'] = 1.0
CENTRAL_HPARAMS['l2_clip'] = 15.0
CENTRAL_HPARAMS['noise_mult'] = 0.01

FED_HPARAMS = {}
FED_HPARAMS['log_freq'] = LOG_FREQ
FED_HPARAMS['batch_size'] = 512
FED_HPARAMS['lr'] = 0.2
FED_HPARAMS['num_epochs'] = 1  # Number of local epochs in each round
FED_HPARAMS['neighborhood_sizes'] = [20]
FED_HPARAMS['focal_alpha'] = FOCAL_ALPHA
FED_HPARAMS['seed'] = 4321
FED_HPARAMS['num_rounds'] = 2
FED_HPARAMS['early_stop'] = 1.0
FED_HPARAMS['mrmtl_lam'] = 0.0003
FED_HPARAMS['l2_clip'] = 15.0
FED_HPARAMS['noise_mult'] = 0.01

DATA_LOSS_FN = partial(focal_loss, alpha=FOCAL_ALPHA)
MODEL_LOSS_FN = partial(forward_focal_loss, alpha=FOCAL_ALPHA)
LOCAL_TESTING = False

if LOCAL_TESTING:
  LOG_FREQ = CENTRAL_HPARAMS['log_freq'] = FED_HPARAMS['log_freq'] = 50

############################################
############### Architecture ###############
############################################


def model_fn(inputs, out_dim=1, num_groups=16, **kwargs):
  """Model definition in JAX/Haiku."""

  ### MLP
  # May use `tanh` instead of `relu` following tempered sigmoids paper
  # May use `groupnorm` instead of any other norm layers following deepmind paper
  # activation_fn = jax.nn.tanh
  # activation_fn = jax.nn.relu
  # return hk.Sequential([
  #     hk.Flatten(),
  #     hk.Linear(512), activation_fn,
  #     hk.GroupNorm(groups=num_groups),
  #     # hk.Linear(384), activation_fn,
  #     # hk.GroupNorm(groups=num_groups),
  #     hk.Linear(256), activation_fn,
  #     hk.GroupNorm(groups=num_groups),
  #     hk.Linear(out_dim)
  # ])(inputs)

  ### Logistic regression
  zero_init = hk.initializers.Constant(0)
  return hk.Sequential([
      hk.Flatten(),
      hk.Linear(out_dim, w_init=zero_init),
  ])(inputs)


############################################
############## Model training ##############
############################################


def create_model(model_fn):
  model = hk.without_apply_rng(hk.transform(model_fn))
  return model


def init_model(model, batch_input_template, seed: int):
  key = random.PRNGKey(seed=seed)
  params = model.init(key, batch_input_template)
  return params


def forward_model(model, params, inputs):
  logits = model.apply(params=params, inputs=inputs).squeeze()
  return logits


def compute_batch_metrics(logits, targets):
  """Compute metrics for the given logits and targets."""
  logits = jnp.atleast_1d(logits)
  targets = jnp.atleast_1d(targets)
  probs = jnp.atleast_1d(jax.nn.sigmoid(logits))
  preds = jnp.round(probs)
  auprc = average_precision_score(targets, probs)
  acc = float(jnp.mean(jnp.round(probs) == targets))
  return auprc, acc


def train_model(model,
                params: chex.ArrayTree,
                loader: tf.data.Dataset,
                num_epochs: int,
                lr: float,
                mrmtl_lam: float = 0.0,
                mrmtl_params: Optional[chex.ArrayTree] = None,
                l2_clip=None,
                noise_mult=None,
                seed=None,
                early_stop: float = 1.0):
  """Train a model on the given dataloader."""

  # Create optimizer (leave out momentum for now)
  using_dpsgd = l2_clip is not None and noise_mult is not None
  if using_dpsgd:
    if seed is None:  # randomly choose a seed if not provided
      rng = np.random.default_rng()
      seed = rng.integers(low=0, high=2**32 - 1)
    logger.info(f'************* Using DPSGD with seed {seed} *************')
    optimizer = optax.dpsgd(learning_rate=lr,
                            l2_norm_clip=l2_clip,
                            noise_multiplier=noise_mult,
                            seed=seed)
  else:
    optimizer = optax.sgd(learning_rate=lr)
  opt_state = optimizer.init(params)

  # Create loss function, and per-example vmapped version
  model_loss_fn = partial(MODEL_LOSS_FN, model)
  lossgrad_fn = vmap(value_and_grad(model_loss_fn), in_axes=(None, 0), out_axes=0)

  @jit
  def batch_update(iter_idx, cur_params, cur_opt_state, batch):
    logits = forward_model(model, cur_params, batch[0])
    example_losses, grads = lossgrad_fn(cur_params, batch)

    if mrmtl_lam > 0.0 and mrmtl_params is not None:
      # Add the model proximal grads to the example grads
      model_diff = tree_map(lambda x, y: x - y, cur_params, mrmtl_params)
      model_diff_grads = tree_map(lambda x: x * mrmtl_lam * 2.0, model_diff)
      grads = tree_map(lambda x, y: x + y, grads, model_diff_grads)

    if not using_dpsgd:
      # Mean over the example gradients
      grads = tree_map(lambda x: jnp.mean(x, axis=0), grads)

    updates, new_opt_state = optimizer.update(grads, cur_opt_state, cur_params)
    new_params = optax.apply_updates(cur_params, updates)
    # Also return logits for logging
    return new_params, new_opt_state, logits, example_losses

  if LOCAL_TESTING:
    logger.remove(0)  # the first handler should be the default one
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)

  num_batches = len(loader)
  logger.info(f'{num_epochs=} * {num_batches=} == {num_epochs * num_batches} iters')
  for epoch_idx in range(num_epochs):
    epoch_tqdm = tqdm(loader.as_numpy_iterator(),
                      total=num_batches,
                      desc=f'Training epoch = {epoch_idx + 1} (update/1min)',
                      mininterval=60.0)
    for batch_idx, batch in enumerate(epoch_tqdm):
      _t0 = time.perf_counter()
      batch = device_put(batch)
      batch_x, batch_y = batch
      iter_idx = epoch_idx * num_batches + batch_idx
      _t1 = time.perf_counter()
      params, opt_state, logits, losses = batch_update(iter_idx, params, opt_state, batch)
      _t2 = time.perf_counter()
      # Metrics
      if iter_idx % LOG_FREQ == 0:
        # auprc, acc, recall, f1, fp, fn = compute_batch_metrics(logits, batch_y)
        auprc, acc = compute_batch_metrics(logits, batch_y)
        loss = float(losses.mean())
        _t3 = time.perf_counter()
        metrics = {
            'iter_idx': iter_idx,
            'batch_auprc': auprc,
            'batch_loss': loss,
            'batch_acc': acc,
            'batch_update_time': _t2 - _t0,
            'batch_metrics_time': _t3 - _t2,
            'batch_total_time': _t3 - _t0,
        }
        logger.info(metrics)

      # Early stopping if needed
      if early_stop < 1.0 and batch_idx > (early_stop * num_batches):
        logger.info(f'Early stopping at {batch_idx=} with {early_stop=}')
        break

    gc.collect()

  logger.info('Finished training!')
  return params


def evaluate_model(model, params, loader):
  epoch_logits, epoch_targets = [], []
  eval_loader = tqdm(loader.as_numpy_iterator(), total=len(loader), desc='Evaluating')
  for batch_idx, batch in enumerate(eval_loader):
    batch = device_put(batch)
    batch_x, batch_y = batch
    logits = forward_model(model, params, batch_x)
    epoch_logits.append(np.asarray(logits))
    epoch_targets.append(np.asarray(batch_y))

  # Epoch metrics
  epoch_logits, epoch_targets = np.concatenate(epoch_logits), np.concatenate(epoch_targets)
  loss = float(np.mean(DATA_LOSS_FN(epoch_logits, epoch_targets)))
  # auprc, acc, recall, f1, fp, fn = compute_batch_metrics(epoch_logits, epoch_targets)
  auprc, acc = compute_batch_metrics(epoch_logits, epoch_targets)
  epoch_preds = np.round(np_sigmoid(epoch_logits))
  recall = recall_score(epoch_targets, epoch_preds, zero_division=0)
  f1 = f1_score(epoch_targets, epoch_preds, zero_division=0)
  _, fp, fn, _ = confusion_matrix(epoch_targets, epoch_preds, labels=[0, 1],
                                  normalize='true').ravel()
  metrics = {
      'eval_auprc': auprc,
      'eval_loss': loss,
      'eval_acc': acc,
      'eval_recall': recall,
      'eval_f1': f1,
      'eval_fp': fp,
      'eval_fn': fn
  }
  logger.info(metrics)


def inference_model(model, params, loader):

  epoch_logits, epoch_pids = [], []
  loader = tqdm(loader.as_numpy_iterator(),
                total=len(loader),
                desc='Inferencing',
                disable=(not LOCAL_TESTING))
  for batch_idx, batch in enumerate(loader):
    batch = device_put(batch)
    batch_x, batch_pids = batch
    logits = forward_model(model, params, batch_x)
    epoch_logits.append(np.asarray(logits))
    epoch_pids.append(np.asarray(batch_pids))

  epoch_logits, epoch_pids = np.concatenate(epoch_logits), np.concatenate(epoch_pids)
  epoch_probs = np_sigmoid(epoch_logits)
  return epoch_probs, epoch_pids


############################################
################# Model IO #################
############################################


def save_params(save_dir: str, params: chex.ArrayTree, fname=None) -> None:
  fname = fname or 'params.npy'
  save_dir = Path(save_dir)
  with open(save_dir / fname, 'wb') as f:
    for x in tree_leaves(params):
      jnp.save(f, x, allow_pickle=False)

  tree_struct = jax.tree_map(lambda t: 0, params)
  with open(save_dir / 'model_tree.pkl', 'wb') as f:
    pickle.dump(tree_struct, f)

  logger.info(f'Params saved to {save_dir.resolve()} with {fname=}')


def load_params(save_dir, fname=None) -> chex.ArrayTree:
  fname = fname or 'params.npy'
  save_dir = Path(save_dir)
  with open(save_dir / 'model_tree.pkl', 'rb') as f:
    tree_struct = pickle.load(f)

  leaves, treedef = tree_flatten(tree_struct)
  with open(save_dir / fname, 'rb') as f:
    flat_params = [jnp.load(f) for _ in leaves]

  params = tree_unflatten(treedef, flat_params)
  logger.info(f'Params loaded from to {save_dir.resolve()} with {fname=}')
  return params


############################################
############## Utils functions #############
############################################


def global_l2_norm_sq(tensor_struct):
  """Computes the global L2 norm squared in an autodiff friendly way.

  You can get NaNs from `jnp.linalg.norm`; the gist is that `norm` is not
  differentiable at 0, but `squared-norm` is indeed differentiable at 0
  (needed for l2 regularization). See:
    https://github.com/google/jax/issues/3058
    https://github.com/google/jax/issues/6484
  """
  flat_vec = struct_flatten(tensor_struct)
  return flat_vec @ flat_vec


def global_l2_norm(tensor_struct):
  """Computes the global L2 norm of tensor tree."""
  return jnp.sqrt(global_l2_norm_sq(tensor_struct))


def array_shapes(struct):
  return tree_map(lambda x: x.shape, struct)


def num_params(struct):
  param_list, _ = tree_flatten(struct)
  return np.sum([w.size for w in param_list])  # Use numpy since shape is static.


def struct_add(struct_1, struct_2):
  return tree_map(jnp.add, struct_1, struct_2)


def struct_sub(struct_1, struct_2):
  return tree_map(jnp.subtract, struct_1, struct_2)


def struct_mul(struct_1, struct_2):
  return tree_map(jnp.multiply, struct_1, struct_2)


def struct_div(struct_1, struct_2):
  return tree_map(jnp.divide, struct_1, struct_2)


def struct_sq(struct):
  return tree_map(jnp.square, struct)


def struct_sqrt(struct):
  return tree_map(jnp.sqrt, struct)


def struct_add_scalar(struct, value):
  t_list, tree_def = tree_flatten(struct)
  new_t_list = [t + value for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


def struct_mul_scalar(struct, factor):
  t_list, tree_def = tree_flatten(struct)
  new_t_list = [t * factor for t in t_list]
  return tree_unflatten(tree_def, new_t_list)


def struct_div_scalar(struct, factor):
  return struct_mul_scalar(struct, 1.0 / factor)


def struct_average(struct_list, weights=None):
  def average_fn(*tensor_list):
    return jnp.average(jnp.asarray(tensor_list), axis=0, weights=weights)

  return tree_map(average_fn, *struct_list)


@jit
def struct_flatten(struct):
  tensors, tree_def = tree_flatten(struct)
  flat_vec = jnp.concatenate([t.reshape(-1) for t in tensors])
  return flat_vec


@jit
def struct_unflatten(flat_vec, struct_template):
  t_list, tree_def = tree_flatten(struct_template)
  pointer, split_list = 0, []
  for tensor in t_list:
    length = np.prod(tensor.shape)  # Shape is static so np is fine
    split_vec = flat_vec[pointer:pointer + length]
    split_list.append(split_vec.reshape(tensor.shape))
    pointer += length
  return tree_unflatten(tree_def, split_list)


def struct_concat(struct_list):
  flat_vecs = [struct_flatten(struct) for struct in struct_list]
  return jnp.concatenate(flat_vecs)


def struct_zeros_like(struct):
  return tree_map(jnp.zeros_like, struct)
