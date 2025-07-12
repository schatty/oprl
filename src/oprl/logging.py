import os
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
import shutil
from abc import ABC, abstractmethod
from typing import Any, Protocol, Callable

import torch as t
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


class LoggerProtocol(Protocol):
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...

    def log_scalars(self, values: dict[str, float], step: int) -> None: ...


def get_logs_path(logdir: str, algo: str, env: str, seed: int) -> Path:
    dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    log_dir = Path(logdir) / algo / f"{algo}-env_{env}-seed_{seed}-{dt}"
    logging.info(f"LOGDIR: {log_dir}")
    return log_dir


def create_stdout_logger(name=None):
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        name = os.path.splitext(filename)[0]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)    
    return logger


def copy_exp_dir(log_dir: Path) -> None:
    cur_dir = Path(__file__).parents[0]
    dest_dir = log_dir / "src"
    shutil.copytree(cur_dir, dest_dir)
    logging.info(f"Source copied into {dest_dir}")


def make_text_logger_func(config: dict, algo, env) -> Callable:
    def make_logger(seed: int) -> LoggerProtocol:
        logs_root = os.environ.get("OPRL_LOGS", "logs")
        log_dir = get_logs_path(logdir=logs_root, algo=algo, env=env, seed=seed)
        logger = FileTxtLogger(log_dir, config)
        logger.copy_source_code()
        return logger
    return make_logger


def save_json_config(config: dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(config, f)


class BaseLogger(ABC):
    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        ...

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        """
        Args:
            values: Dict with tag -> value to log.
            step: Iter step.
        """
        (self.log_scalar(k, v, step) for k, v in values.items())


class FileTxtLogger(BaseLogger):
    def __init__(self, logdir: Path | str, config: dict[str, Any]) -> None:
        self.writer = SummaryWriter(logdir)
        self.log_dir = Path(logdir)
        self.config = config

    def copy_source_code(self) -> None:
        copy_exp_dir(self.log_dir)
        logging.info(f"Source code is copied to {self.log_dir}.")
        save_json_config(self.config, self.log_dir / "config.json")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        self._log_scalar_to_file(tag, value, step)

    def save_weights(self, weights: nn.Module, step: int) -> None:
        weights_path = self.log_dir / "weights" / f"{step}.w"
        weights_path.parents[0].mkdir(exist_ok=True)
        t.save(
            weights,
            weights_path
        )

    def _log_scalar_to_file(self, tag: str, val: float, step: int) -> None:
        log_path = self.log_dir / f"{tag}.log"
        log_path.parents[0].mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{step} {val}\n")



