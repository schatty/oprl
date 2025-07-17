from typing import Callable
from multiprocessing import Process

from oprl.algos.protocols import AlgorithmProtocol, PolicyProtocol
from oprl.buffers.protocols import ReplayBufferProtocol
from oprl.environment.protocols import EnvProtocol
from oprl.logging import LoggerProtocol, create_stdout_logger
from oprl.runners.config import DistribConfig


logger = create_stdout_logger()


def run_distrib_training(
    run_env_worker: Callable,
    run_policy_update_worker: Callable,
    make_env: Callable[[int], EnvProtocol],
    make_algo: Callable[[LoggerProtocol], AlgorithmProtocol],
    make_policy: Callable[[], PolicyProtocol],
    make_replay_buffer: Callable[[], ReplayBufferProtocol],
    make_logger: Callable[[], LoggerProtocol],
    config: DistribConfig
) -> None:
    processes = []
    for i_env in range(config.num_env_workers):
        processes.append(
            Process(target=run_env_worker, args=(make_env, make_policy, config, i_env))
        )
    processes.append(
        Process(
            target=run_policy_update_worker,
            args=(make_algo, make_env, make_replay_buffer, make_logger, config),
        )
    )

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    logger.info("Training Finished.")
