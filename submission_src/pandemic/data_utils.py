"""This files handles data loading from CSV files."""
import os
# Just use CPU for now as the model is quite small
os.environ['CPU_OR_GPU'] = 'cpu'  # Submission specific envvar
import gc  # Try manual garbage collection to help with memory issues

import h5py
from loguru import logger
import graph_tool as gt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
import tensorflow as tf  # Only used for tf.data
from scipy.stats import boxcox

from sklearn.preprocessing import StandardScaler

from src.graph_utils import parse_contact_graph

############# Constants relating to the data & preprocessing #############
PID_PLACEHOLDER = IDX_PLACEHOLDER = -1
WINDOW_LEN = 21  # Number of past days to use for features
PRED_LEN = 7  # Number of future days to predict
NUM_DAYS = 57
INFECTION_STATE = {'S': 0, 'I': 1, 'R': 2}
NUM_STATES = len(INFECTION_STATE)
NUM_ACTIVITIES = 7
ENCODE_ACTIVITY = False

D_PERSONAL = 8  # see `data_utils.create_node_features`
D_CENTRALITY = 2  # deg, katz + (pagerank + eigenvalue)
D_HISTORY = 21 * 3  # 21 days, 3 features each
D_NODE = D_CENTRALITY + D_PERSONAL + D_HISTORY
USE_FULL_LOC = True
if USE_FULL_LOC:
  D_LOC = 8  # long/lat/work/school/religion/other/college/shopping
else:
  D_LOC = 4  # long/lat/work/school

if ENCODE_ACTIVITY:
  D_EDGE = 1 + 1 + NUM_ACTIVITIES * 2 + D_LOC
else:
  D_EDGE = 1 + 1 + D_LOC

###########################################
################## Utils ##################
###########################################


def read_csv_helper(path,
                    indices: List[str],
                    check_unique: bool = True,
                    sort_method=None,
                    usecols: Optional[List[str]] = None):
  logger.info(f'Loading {Path(path).resolve()} ...')
  df = pd.read_csv(path, usecols=usecols, index_col=indices, low_memory=True)
  if check_unique:
    assert df.index.is_unique, f'{path=} has duplicate indices'
  # Sort all indices for faster query
  # Stable-sort may use O(n) memory as opposed to O(log n) by quicksort
  if sort_method:
    assert sort_method in ['stable', 'quicksort'], f'unknown {sort_method=}'
    df.sort_index(inplace=True, kind=sort_method)
  return df


def array_save(arr: np.ndarray, fname: str, cache_dir: Path, client_id: Optional[str] = None):
  """Cache features to disk."""
  logger.info(f'[{client_id=}] Caching {fname=} to {cache_dir=} ...')
  cache_dir.mkdir(parents=True, exist_ok=True)
  path = cache_dir / f'cache_client_{client_id}_{fname}.h5'
  with h5py.File(path, 'w') as hf:
    hf.create_dataset(fname, data=arr, compression='gzip')


def array_load(fname: str, cache_dir: Path, client_id: Optional[str] = None) -> np.ndarray:
  logger.info(f'[{client_id=}] Loading {fname=} from {cache_dir=} ...')
  path = cache_dir / f'cache_client_{client_id}_{fname}.h5'
  with h5py.File(path, 'r') as hf:
    arr = hf[fname][:]
  return arr


def array_exists(fname: str, cache_dir: Path, client_id: Optional[str] = None) -> bool:
  path = cache_dir / f'cache_client_{client_id}_{fname}.h5'
  return path.exists()


def features_cached(cache_dir: Path, client_id: Optional[str] = None) -> bool:
  return all([
      array_exists('idx2pid', cache_dir, client_id),
      array_exists('idx2nabes_idx', cache_dir, client_id),
      array_exists('idx2edgefeats', cache_dir, client_id),
      array_exists('idx2nodefeats', cache_dir, client_id),
      array_exists('idx2outcome', cache_dir, client_id),
  ])


################################################################################
################## Feature construction in TensorFlow / NumPy ##################
################################################################################


def preprocess_outcome_df(df_path: Path, client_id=None) -> np.ndarray:
  logger.info(f'[{client_id=}] Preprocessing outcomes DF for features ...')
  df = read_csv_helper(df_path, ['pid'], check_unique=False)
  assert df.index.name == 'pid', f'Dataframe must be indexed by pid, got {df.index.name=}'
  #### Memory efficient implementation assumes days and pids are sorted
  df.drop(columns=['day'], inplace=True)
  logger.info(f'[{client_id=}] dropped days from outcomes ...')
  df.replace({'state': INFECTION_STATE}, inplace=True)
  logger.info(f'[{client_id=}] replaced state in outcomes ...')
  arr = df.to_numpy().reshape(NUM_DAYS, -1).T  # Need to transpose
  assert arr.shape[1] == NUM_DAYS, f'Expected {NUM_DAYS} days, got {arr.shape[1]}'
  return arr


def labeling_fn(window_states: np.ndarray, future_states: np.ndarray):
  cond_1 = np.all(window_states == INFECTION_STATE['S'])
  cond_2 = np.logical_or(np.any(future_states == INFECTION_STATE['I']),
                         np.any(future_states == INFECTION_STATE['R']))
  label = np.logical_and(cond_1, cond_2).astype(np.float32)
  return label


def create_neighborhood_features(nabes_idx: np.ndarray, edge_feats: np.ndarray,
                                 idx2nodefeats: np.ndarray, idx2outcome: np.ndarray,
                                 start_day: np.ndarray):
  """
  Create feature vectors from a list of self & neighbor `pid`s (including self id).

  Args:
    nabes_idx: (n_nodes, n_nabes + 1), each row is a neighborhood of nodes
    edge_feats: (n_nodes, n_nabes + 1, d_edge), edge features for each node
  """
  valid_mask = (nabes_idx != IDX_PLACEHOLDER).astype(np.float32)  # (n_nabes+1, ) #
  nabe_feats = idx2nodefeats[nabes_idx].astype(np.float32)  # (n_nabes+1, k + d)
  n_pre_history_feats = nabe_feats.shape[1]

  # Obtain the infection outcome for all nodes, slice the window, and one-hot encode
  nabe_outcomes = idx2outcome[nabes_idx]  # (n_nabes+1, n_days)
  nabe_windows = nabe_outcomes[:, start_day:start_day + WINDOW_LEN]  # (n_nabes+1, window_len)
  nabe_onehot_states = np.eye(NUM_STATES)[nabe_windows]  # (n_nabes+1, window_len, n_states)

  nabe_onehot_states = np.reshape(nabe_onehot_states, [-1, WINDOW_LEN * NUM_STATES])
  # Append the one-hot encoded states to the feature vector
  nabe_feats = np.concatenate([nabe_feats, nabe_onehot_states], axis=1)  # (n_nabes+1, n_feats)

  # Since we had placeholders, we need to mask the features to 0
  nabe_feats = valid_mask[:, None] * nabe_feats.astype(np.float32)  # (n_nabes+1, n_feats)
  # Remove the infection states from self node (since it's either all S or not)
  self_feats = nabe_feats[0][:n_pre_history_feats]  # (n_feats,)
  # Add binary indicator for whether self node had all S in the window
  self_all_S = np.all(nabe_windows[0] == INFECTION_STATE['S']).astype(np.float32)
  self_feats = np.concatenate([self_feats, [self_all_S]], axis=0)  # (n_nodefeats + 1,)

  # Concat edge features # (n_nabes, n_feats + n_edgefeats)
  hop_feats = np.concatenate([nabe_feats[1:], edge_feats], axis=1)
  # Flatten all neighbor features
  hop_feats = hop_feats.ravel()  # (n_nabes * n_feats)

  nabe_feats = np.concatenate([self_feats, hop_feats], axis=0)
  return nabe_feats


def create_training_example(nabes_idx: np.ndarray, edge_feats: np.ndarray,
                            idx2nodefeats: np.ndarray, idx2outcome: np.ndarray):
  maxval = NUM_DAYS - WINDOW_LEN - PRED_LEN
  rng = np.random.default_rng()
  start_day = rng.integers(size=(), low=0, high=maxval, dtype=np.int32)  # seedless random
  nabe_feats = create_neighborhood_features(nabes_idx, edge_feats, idx2nodefeats, idx2outcome,
                                            start_day)
  self_states = idx2outcome[nabes_idx[0]]  # (n_day)
  self_window_states = self_states[start_day:(start_day + WINDOW_LEN)]
  self_future_states = self_states[(start_day + WINDOW_LEN):(start_day + WINDOW_LEN + PRED_LEN)]

  label = labeling_fn(self_window_states, self_future_states)
  return nabe_feats, label


def create_prediction_input(nabes_idx: np.ndarray, edge_feats: np.ndarray, idx2pid: np.ndarray,
                            idx2nodefeats: np.ndarray, idx2outcome: np.ndarray):
  """Create a prediction input from a list of neighbor `pid`s (including self)."""
  # Create neighborhood features from the last window
  start_day = NUM_DAYS - WINDOW_LEN
  nabe_feats = create_neighborhood_features(nabes_idx, edge_feats, idx2nodefeats, idx2outcome,
                                            start_day)
  self_pid = idx2pid[nabes_idx[0]]
  return nabe_feats, self_pid


def create_node_features(person_data_path: Path,
                         household_data_path: Path,
                         activity_location_assignment_data_path: Path,
                         client_id=None):

  logger.info(f'[{client_id=}] Building node features ...')
  person_df = read_csv_helper(person_data_path, ['pid'], sort_method='stable')
  household_df = read_csv_helper(household_data_path, ['hid'], sort_method='stable')
  ala_df = read_csv_helper(activity_location_assignment_data_path, ['pid', 'lid'],
                           check_unique=False,
                           sort_method='quicksort')

  ### Person: drop person_number, normalize age and sex
  assert person_df.index.name == 'pid', f'Dataframe must be indexed by pid, got {person_df.index.name=}'
  # feats_df = person_df.drop(columns=['person_number', 'hid'])
  feats_df = person_df.drop(columns=['person_number'])
  feats_df['age'] = StandardScaler().fit_transform(feats_df['age'].to_numpy().reshape(-1, 1))
  feats_df['sex'] -= 1.5  # shift {1, 2} to {-0.5, 0.5}

  ### Household: add household size to features
  assert household_df.index.name == 'hid', f'Dataframe must be indexed by hid, got {household_df.index.name=}'
  # Merge with person_df and assign hh_size to person_df
  feats_df = feats_df.merge(household_df[['hh_size']], left_on='hid', right_index=True)
  # Apply box-cox transform to hh_size
  feats_df['hh_size'] = boxcox(feats_df['hh_size'])[0]

  ### Activity binary indicators
  # Work=2: whether the person has ever been to work
  pids_worked = ala_df[ala_df.activity_type == 2].index.get_level_values(0).unique()
  # School=5, college=6: whether the person has ever been to school
  pids_schooled = ala_df[ala_df.activity_type.isin([5, 6])].index.get_level_values(0).unique()
  # Shopping=3: whether the person has ever been to shopping
  pids_shopped = ala_df[ala_df.activity_type == 3].index.get_level_values(0).unique()
  # Home=1: whether the person has ALWAYS stayed at home; use `np.prod` to speed things up
  pids_homeonly = ala_df.groupby('pid').activity_type.agg(np.prod) == 1
  # binary features in {-0.5, 0.5}
  feats_df['worked'] = feats_df.index.isin(pids_worked).astype(float) - 0.5
  feats_df['schooled'] = feats_df.index.isin(pids_schooled).astype(float) - 0.5
  feats_df['shopped'] = feats_df.index.isin(pids_shopped).astype(float) - 0.5
  feats_df['homeonly'] = pids_homeonly.to_numpy().astype(float) - 0.5

  ### Fraction of day that a person has spent outside (at work, school, shopping)
  out_activities = [2, 3, 5, 6]
  out_time = ala_df[ala_df.activity_type.isin(out_activities)].groupby('pid').duration.agg('sum')
  feats_df['out_time'] = out_time / 86_400  # convert to fraction of day
  feats_df['out_time'] = feats_df['out_time'].fillna(0)  # fill NaN with 0

  # Drop useless IDs
  feats_df = feats_df.drop(columns=['hid'])
  logger.info(f'[{client_id=}] Done building node features')
  return feats_df.to_numpy()


def create_graph_features(
    # Take file paths and dump CSVs/DFs at the end of the function
    person_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    population_network_data_path: Path,
    # Parameters for features
    neighborhood_sizes: List[int],
    client_id: Optional[str] = None,
):
  logger.info(
      f'[{client_id=}] create_graph_features paths: {person_data_path=}, {residence_location_data_path=} {activity_location_data_path=} {population_network_data_path=} ...'
  )
  logger.info(f'[{client_id=}] Creating edge features ... ')
  # Returns: idx2centrality, idx2nabes, idx2edgefeats
  assert len(neighborhood_sizes) == 1, f'Supports 1-hop neighborhood for now'
  num_1hop = neighborhood_sizes[0]
  # Calls graph-dependent functions separately so that this context is free of `graph` object
  idx2pid, idx2nabes_idx, idx2edgefeats, idx2centrality = parse_contact_graph(
      person_data_path, population_network_data_path, num_1hop, client_id)
  gc.collect()

  logger.info(f'[{client_id=}] Preprocessing start/duration ...')
  # idx2edgefeats is currently (n_nodes, n_1hop, 5).
  # This normalizes the start_time and duration
  f_start_duration = idx2edgefeats[:, :, [1, 2]] / 86400.0

  # Convert act1 and act2 to one-hot for 6 types of activities
  if ENCODE_ACTIVITY:
    logger.info(f'[{client_id=}] Preprocessing act1/act2 ...')
    f_act1 = np.eye(NUM_ACTIVITIES)[idx2edgefeats[:, :, 3]]
    f_act2 = np.eye(NUM_ACTIVITIES)[idx2edgefeats[:, :, 4]]

  f_lid = idx2edgefeats[:, :, 0].reshape(-1)
  del idx2edgefeats
  gc.collect()

  # Read location features
  logger.info(f'[{client_id=}] Reading residence location ...')
  rl_df = read_csv_helper(residence_location_data_path,
                          indices=['rlid'],
                          sort_method='quicksort',
                          usecols=['rlid', 'longitude', 'latitude'])
  # Merge residence and activity location
  logger.info(f'[{client_id=}] Preprocessing residence location ...')
  rl_df.index.rename('lid', inplace=True)

  # Different kinds of location features (residence & activity locations)
  if USE_FULL_LOC:
    rl_df[['work', 'shopping', 'school', 'other', 'college', 'religion']] = 0
    al_cols = [
        'alid', 'longitude', 'latitude', 'work', 'shopping', 'school', 'other', 'college',
        'religion'
    ]
  else:
    rl_df[['work', 'school']] = 0
    al_cols = ['alid', 'longitude', 'latitude', 'work', 'school', 'college']

  logger.info(f'[{client_id=}] Reading activity location ...')
  al_df = read_csv_helper(activity_location_data_path,
                          indices=['alid'],
                          sort_method='quicksort',
                          usecols=al_cols)
  logger.info(f'[{client_id=}] Preprocessing activity location ...')
  al_df.index.rename('lid', inplace=True)

  if not USE_FULL_LOC:
    # Only keep work + school, and merge school and college
    al_df['school'] = al_df['school'] | al_df['college']
    al_df = al_df.drop(columns=['college'])

  logger.info(f'[{client_id=}] Concat locations ...')
  loc_df = pd.concat([rl_df, al_df], axis=0).sort_index()
  logger.info(f'[{client_id=}] Standardizing long/lat ...')
  scaler = StandardScaler()
  loc_df[['longitude', 'latitude']] = scaler.fit_transform(loc_df[['longitude', 'latitude']])

  # Replace the lid with relevant features from the res_loc_df or act_loc_df
  # Note that we have -1s, so we need to replace them with any valid lids
  logger.info(f'[{client_id=}] Creating location features ...')
  f_lid[f_lid == -1] = loc_df.index[0]
  # (n_nodes, n_1hop, loc_feats)

  #  We can use `loc_df.reindex(f_lid)` to fill in the missing values with NaNs.
  logger.info(f'[{client_id=}] Indexing location features ...')
  f_lid = loc_df.reindex(f_lid, method='nearest')
  f_lid = f_lid.to_numpy().reshape(-1, num_1hop, loc_df.shape[1])

  del rl_df, al_df, loc_df
  gc.collect()
  # - concatenate all the features
  logger.info(f'[{client_id=}] Merging edge features ...')
  if ENCODE_ACTIVITY:
    idx2edgefeats = np.concatenate([f_start_duration, f_act1, f_act2, f_lid], axis=-1)
  else:
    idx2edgefeats = np.concatenate([f_start_duration, f_lid], axis=-1)

  idx2edgefeats = idx2edgefeats.astype(np.float32)
  logger.info(f'[{client_id=}] Done creating edge features ')
  assert idx2edgefeats.shape[-1] == D_EDGE, f'Unexpected nfeats={idx2edgefeats.shape[-1]}'
  return idx2pid, idx2nabes_idx, idx2edgefeats, idx2centrality


###################################################
################# Main entrypoint #################
###################################################


def create_loader(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    batch_size: int,
    is_prediction: bool,
    neighborhood_sizes: List[int] = [30],
    disease_outcome_target_path: Optional[Path] = None,
    client_id: Optional[str] = None,
    return_num_examples: bool = False,
    cache_dir: Optional[Path] = None,
) -> None:
  """Same loader as before, except we incorporate edge features, and take data paths as needed."""
  assert len(neighborhood_sizes) == 1, 'Only support 1-hop neighbors for now'
  # Caching:
  # - To save: idx2nabes_idx, idx2edgefeats, idx2pid, idx2nodefeats, idx2outcome,
  # - No need: idx2centrality, idx2personal
  # Check randomness in generating the above: everything is deterministic
  logger.info(f'[{client_id=}] Cache features? {cache_dir=}')

  if (cache_dir is not None) and features_cached(cache_dir, client_id):
    logger.info(f'[{client_id=}] ************* Loading cached features *************')
    idx2pid = array_load('idx2pid', cache_dir, client_id)
    idx2nabes_idx = array_load('idx2nabes_idx', cache_dir, client_id)
    idx2edgefeats = array_load('idx2edgefeats', cache_dir, client_id)
    idx2nodefeats = array_load('idx2nodefeats', cache_dir, client_id)
    idx2outcome = array_load('idx2outcome', cache_dir, client_id)

  else:
    logger.info(f'[{client_id=}] Generating fresh features ...')
    # Do feature generation as usual
    logger.info(f'[{client_id=}] Preprocessing outcomes DF ...')
    idx2outcome = preprocess_outcome_df(disease_outcome_data_path,
                                        client_id=client_id)  # (n_nodes, days)
    gc.collect()

    logger.info(f'[{client_id=}] Creating graph-based features ...')
    idx2pid, idx2nabes_idx, idx2edgefeats, idx2centrality = create_graph_features(
        person_data_path,
        residence_location_data_path,
        activity_location_data_path,
        population_network_data_path,
        neighborhood_sizes,
        client_id,
    )
    gc.collect()

    logger.info(f'[{client_id=}] Building node features ...')
    idx2personal = create_node_features(person_data_path,
                                        household_data_path,
                                        activity_location_assignment_data_path,
                                        client_id=client_id)  # (n_nodes, d)
    idx2nodefeats = np.concatenate([idx2centrality, idx2personal], axis=1)  # (n_nodes, d)
    del idx2centrality, idx2personal
    gc.collect()

    if cache_dir is not None:
      logger.info(f'[{client_id=}] ************* Caching features to disk ************* ')
      array_save(idx2pid, 'idx2pid', cache_dir, client_id)
      array_save(idx2nabes_idx, 'idx2nabes_idx', cache_dir, client_id)
      array_save(idx2edgefeats, 'idx2edgefeats', cache_dir, client_id)
      array_save(idx2nodefeats, 'idx2nodefeats', cache_dir, client_id)
      array_save(idx2outcome, 'idx2outcome', cache_dir, client_id)

  logger.info(f'[{client_id=}] Creating dataloader ...')
  num_nodes = len(idx2pid)
  # Start from the neighborhood directly: (n_nodes, n_1hop_nabes + 1) and (n_nodes, n_1hop_nabes, 22)
  loader = tf.data.Dataset.from_tensor_slices((idx2nabes_idx, idx2edgefeats))

  if is_prediction:
    _create_input = lambda nabes_idx, edge_feats: create_prediction_input(
        nabes_idx, edge_feats, idx2pid, idx2nodefeats, idx2outcome)
  else:
    _create_input = lambda nabes_idx, edge_feats: create_training_example(
        nabes_idx, edge_feats, idx2nodefeats, idx2outcome)

  # Shuffle `pid`s (so neighborhoods) for each epoch
  # Remove seed for shuffling for now, since window sampling is seedless
  if not is_prediction:
    loader = loader.shuffle(buffer_size=num_nodes, reshuffle_each_iteration=True)

  ret_type = (tf.float32, tf.int64) if is_prediction else (tf.float32, tf.float32)
  loader = loader.map(lambda nabes_idx, edge_feats: tf.numpy_function(
      _create_input, [nabes_idx, edge_feats], ret_type),
                      num_parallel_calls=tf.data.AUTOTUNE)

  loader = loader.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  if return_num_examples:
    return loader, num_nodes

  return loader
