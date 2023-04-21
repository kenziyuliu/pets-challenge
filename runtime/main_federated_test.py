import os
import sys
from pathlib import Path
from typing import Callable

import flwr as fl

from loguru import logger
from supervisor import (
    FederatedSupervisor,
    FederatedWrapperStrategy,
    wrap_test_client_factory,
)

from src.solution_federated import test_client_factory, test_strategy_factory


if __name__ == "__main__":
    logger.info(f"Starting {__file__}...")

    supervisor = FederatedSupervisor(partition_config_path=Path(sys.argv[1]))

    wrapped_client_factory = wrap_test_client_factory(test_client_factory, supervisor)
    solution_strategy, num_rounds = test_strategy_factory(
        server_dir=supervisor.get_server_state_dir()
    )
    wrapped_strategy = FederatedWrapperStrategy(
        solution_strategy=solution_strategy, supervisor=supervisor
    )
    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

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

    # Post-validation
    logger.info("Validating that all required predictions files exist...")
    for cid in supervisor.get_client_ids():
        if supervisor.get_predictions_dest_path(cid=cid):
            assert supervisor.get_predictions_dest_path(cid=cid).exists(), cid
