import os
import sys
from pathlib import Path
from typing import Callable

import flwr as fl

from loguru import logger
from supervisor import (
    FederatedSupervisor,
    FederatedWrapperStrategy,
    wrap_train_client_factory,
)

import src.solution_federated as solution_federated


if __name__ == "__main__":
    logger.info(f"Starting {__file__}...")

    # Pre-validation
    logger.info("Validating that all required functions exist in solution_federated...")
    assert hasattr(solution_federated, "train_client_factory") and callable(
        solution_federated.train_client_factory
    )
    assert hasattr(solution_federated, "train_strategy_factory") and callable(
        solution_federated.train_strategy_factory
    )
    assert hasattr(solution_federated, "test_client_factory") and callable(
        solution_federated.test_client_factory
    )
    assert hasattr(solution_federated, "test_strategy_factory") and callable(
        solution_federated.test_strategy_factory
    )

    supervisor = FederatedSupervisor(partition_config_path=Path(sys.argv[1]))

    # See `supervisor` notes; we wont need this for local testing
    # directly using `solution_federated.train_client_factory` shoudl be enough
    # for data paths, maintain a dictioanry of client_id -> data paths
    wrapped_client_factory = wrap_train_client_factory(
        solution_federated.train_client_factory, supervisor
    )
    solution_strategy, num_rounds = solution_federated.train_strategy_factory(
        server_dir=supervisor.get_server_state_dir()
    )

    # This wrapper is NOT needed for local testing
    wrapped_strategy = FederatedWrapperStrategy(
        solution_strategy=solution_strategy, supervisor=supervisor
    )
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    # For evaluation, we do similar `start_simulation` call for
    # test_client_factory and test_strategy_factory
    # start simulation
    fl.simulation.start_simulation(
        client_fn=wrapped_client_factory,
        clients_ids=supervisor.get_client_ids(),
        client_resources={
            "num_cpus": os.cpu_count() - 1,
        },
        config=server_config,
        strategy=wrapped_strategy,
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
        },
    )
