import gc
from typing import List, Optional
import pandas as pd
from pathlib import Path
import multiprocess as mp  # Use the drop in replacement for multiprocessing
import numpy as np
import graph_tool as gt
from graph_tool.centrality import pagerank, katz, eigenvector
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


def read_csv_helper(path, indices: List[str], check_unique: bool = True, sort_method=None):
  logger.info(f'Loading {Path(path).resolve()} ...')
  df = pd.read_csv(path)
  # See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html
  df.set_index(indices, inplace=True, drop=True, verify_integrity=check_unique)
  # Sort all indices for faster query
  if sort_method:
    assert sort_method in ['stable', 'quicksort'], f'unknown {sort_method=}'
    df.sort_index(inplace=True, kind=sort_method)
  return df


def centrality_measures(graph: gt.Graph):
  num_nodes = graph.num_vertices()
  logger.info('Computing node degrees...')
  idx2deg = graph.get_total_degrees(np.arange(num_nodes))
  # logger.info('Computing katz index...')  NOTE: can have np.nan!
  # idx2katz = np.array(katz(graph).a)
  logger.info('Computing undirected pagerank...')
  idx2pagerank = np.array(pagerank(graph).a)
  # idx2eig could be slow with large graphs
  # logger.info('Computing eigenvector centrality...')
  # _, idx2eig = eigenvector(graph)
  # idx2eig = np.array(idx2eig.a)
  logger.info('Done computing centrality measures')
  # Concat selected features
  idx2centrality = np.vstack([idx2deg, idx2pagerank]).T
  return idx2centrality


def construct_graph(person_df, network_df) -> gt.Graph:
  # While only `network_df` is sufficient for constructing the graph,
  # we may also need `person_df` to map node indices to person IDs.
  # Construct vectorized lookups
  assert person_df.index.is_unique, f'person_df should have unique indices'
  pids = idx2pid = person_df.index.values
  pid2idx = np.ones(max(pids) + 1, dtype=int) * -1
  for idx, pid in enumerate(pids):
    pid2idx[pid] = idx

  logger.info(f'graph: num nodes (pids) = {len(pids)}')
  logger.info(f'graph: num edges = {len(network_df)}')
  # Filter edges among only the provided persons
  pid1s = network_df.index.get_level_values(0)  # Series like object
  pid2s = network_df.index.get_level_values(1)
  # Create grpah nodes
  graph = gt.Graph(directed=False)
  graph.add_vertex(len(pids))
  # Create graph edges
  idx1s = pid2idx[pid1s.values]
  idx2s = pid2idx[pid2s.values]
  edge_list = np.vstack([idx1s, idx2s]).T
  graph.add_edge_list(edge_list)
  logger.info(f'Done generating graph; {graph.num_vertices()=}, {graph.num_edges()=}')
  return graph, idx2pid, pid2idx


def construct_graph_with_edgefeats(
    person_data_path: Path,
    population_network_data_path: Path,
):
  """Like the above, but encodes the edge features as part of the graph."""
  # Read DFs that have unique indices: person, residence loc, activity loc
  person_df = read_csv_helper(person_data_path, indices=['pid'], sort_method='quicksort')

  # Read raw network DF
  logger.info(f'Reading raw network data from {population_network_data_path=}')
  network_df = pd.read_csv(population_network_data_path)

  # Construct index to pid lookups
  pids = idx2pid = person_df.index.values
  pid2idx = np.ones(max(pids) + 1, dtype=int) * -1
  for idx, pid in enumerate(pids):
    pid2idx[pid] = idx

  logger.info(f'graph: num nodes (pids) = {len(pids)}')
  logger.info(f'graph: num edges = {len(network_df)}')

  # Create graph
  graph = gt.Graph(directed=False)
  graph.add_vertex(len(pids))
  # Convert pids to index, before adding edges
  network_df.pid1 = pid2idx[network_df.pid1]
  network_df.pid2 = pid2idx[network_df.pid2]
  # The network_df contains pid1, pid2, lid, start_time, duration, act1, act2, in this order
  edge_list = network_df.values
  ep_lid = graph.new_ep('int64_t')
  ep_start = graph.new_ep('int64_t')
  ep_duration = graph.new_ep('int64_t')
  ep_act1 = graph.new_ep('int64_t')
  ep_act2 = graph.new_ep('int64_t')
  # The only way to ensure edge properties are in the same order as the edges
  # is to create the edges with the property values at the same time.
  # We will keep everything in ints when adding edges, and convert to floats later.
  graph.add_edge_list(edge_list, eprops=[ep_lid, ep_start, ep_duration, ep_act1, ep_act2])
  logger.info(f'Done generating graph; {graph.num_vertices()=}, {graph.num_edges()=}')

  # Derement activity IDs by 1, to make them 0-indexed
  ep_act1.a -= 1
  ep_act2.a -= 1
  return graph, idx2pid, ep_lid, ep_start, ep_duration, ep_act1, ep_act2


def sample_neighborhood_with_edgefeats(graph: gt.Graph, num_1hop: int, ep_lid: gt.EdgePropertyMap,
                                       ep_start: gt.EdgePropertyMap,
                                       ep_duration: gt.EdgePropertyMap, ep_act1: gt.EdgePropertyMap,
                                       ep_act2: gt.EdgePropertyMap):

  placeholder = -1
  n_edge_indicators = 5  # the number of features will grow as we join with other CSVs
  sampled_neighborhood_list = []
  edgefeats_list = []

  # NOTE: Sequentially process the nodes for now!! If compute resources permit
  # one can also parallelize this over read-only graphs.
  indices = np.arange(graph.num_vertices())
  for idx in tqdm(indices, mininterval=60, desc='(every 1min)'):
    nabes_idx_1hop = graph.get_all_neighbors(idx)
    num_nabes = len(nabes_idx_1hop)

    # Handle the case of no neighbors
    if num_nabes == 0:
      sampled_neighborhood_list.append(np.ones(num_1hop, dtype=int) * placeholder)
      edgefeats_list.append(np.ones((num_1hop, n_edge_indicators), dtype=int) * placeholder)
      continue

    rng = np.random.default_rng(seed=idx)
    num_1hop_draw = min(num_1hop, num_nabes)
    rand_nabes = rng.choice(num_nabes, num_1hop_draw, replace=False)

    # Take random 1-hop neighbors, fill with placeholders.
    # Note that this is not the same as actually pruning the graph to, say, bound
    # the maximum node degree, but the difference likely isn't significant in
    # practice since our solution deploys noisy training as an empirical defense
    # instead of a theoretically strong privacy guarantee specifically for the challenge.
    missed = num_1hop - num_1hop_draw
    nabes_idx_1hop = nabes_idx_1hop[rand_nabes]
    sampled_idx_1hop = np.pad(nabes_idx_1hop, (0, missed),
                              constant_values=placeholder)  # (num_1hop,)
    sampled_neighborhood_list.append(sampled_idx_1hop)

    # Take corresponding edges and record edge (contact) features
    hash_edges_1hop = [hash(graph.edge(idx, nabe)) for nabe in nabes_idx_1hop]  # NOTE: bottleneck
    lid_1hop = ep_lid.a[hash_edges_1hop]
    start_1hop = ep_start.a[hash_edges_1hop]
    duration_1hop = ep_duration.a[hash_edges_1hop]
    act1_1hop = ep_act1.a[hash_edges_1hop]
    act2_1hop = ep_act2.a[hash_edges_1hop]
    edgefeats = np.array([lid_1hop, start_1hop, duration_1hop, act1_1hop, act2_1hop]).T
    edgefeats = np.pad(edgefeats, ((0, missed), (0, 0)), constant_values=placeholder)
    edgefeats_list.append(edgefeats)

  # (n_indices, n_1hop) -> (n_indices, n_1hop + 1) to include the node itself
  idx2nabes_idx = np.array(sampled_neighborhood_list)
  idx2nabes_idx = np.concatenate([indices.reshape(-1, 1), idx2nabes_idx], axis=1)
  idx2edgefeats = np.array(edgefeats_list)  # (n_indices, n_1hop, 5)
  return idx2nabes_idx, idx2edgefeats


def parse_contact_graph(person_data_path,
                        population_network_data_path,
                        num_1hop: int,
                        client_id: Optional[str] = None):
  # Construct graph with edge features
  logger.info(f'[{client_id=}] Building graph with edge properties ...')
  graph, idx2pid, ep_lid, ep_start, ep_duration, ep_act1, ep_act2 = construct_graph_with_edgefeats(
      person_data_path, population_network_data_path)
  gc.collect()

  # Extract feature tables as np arrays
  logger.info(f'[{client_id=}] Sampling 1-hop nabes ({num_1hop=}) and querying edge features ...')
  idx2nabes_idx, idx2edgefeats = sample_neighborhood_with_edgefeats(graph, num_1hop, ep_lid,
                                                                    ep_start, ep_duration, ep_act1,
                                                                    ep_act2)
  gc.collect()
  logger.info(f'[{client_id=}] Computing node centrality ...')
  idx2centrality = centrality_measures(graph)  # (n_nodes, k) for k measures
  idx2centrality = MinMaxScaler().fit_transform(idx2centrality)  # per-column normalization
  return idx2pid, idx2nabes_idx, idx2edgefeats, idx2centrality
