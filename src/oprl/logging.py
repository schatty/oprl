import os
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


def create_logdir(logdir: str, algo: str, env: str, seed: int) -> str:
    dt = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    log_dir = os.path.join(logdir, algo, f"{algo}-env_{env}-seed_{seed}-{dt}")
    logging.info(f"LOGDIR: {log_dir}")
    return log_dir


def set_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(filename)s:%(lineno)d\t %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def copy_exp_dir(log_dir: str) -> None:
    cur_dir = os.path.join(os.getcwd(), "src")
    dest_dir = os.path.join(log_dir, "src")
    shutil.copytree(cur_dir, dest_dir)
    logging.info(f"Source copied into {dest_dir}")


def make_text_logger_func(config: dict, algo, env) -> Callable:
    def make_logger(seed: int) -> LoggerProtocol:
        logs_root = os.environ.get("OPRL_LOGS", "logs")
        log_dir = create_logdir(logdir=logs_root, algo=algo, env=env, seed=seed)
        logger = FileTxtLogger(log_dir, config)
        logger.copy_source_code()
        return logger
    return make_logger


def save_json_config(config: dict[str, Any], path: str):
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


class StdLogger(BaseLogger):
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        logging.info(f"{tag}\t{value}\tat step {step}")


class FileTxtLogger(BaseLogger):
    def __init__(self, logdir: str, config: dict[str, Any]) -> None:
        self.writer = SummaryWriter(logdir)
        self.log_dir = logdir
        self.config = config

    def copy_source_code(self) -> None:
        logging.info(f"Source code is copied to {self.log_dir}")
        copy_exp_dir(self.log_dir)
        save_json_config(self.config, os.path.join(self.log_dir, "config.json"))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        self._log_scalar_to_file(tag, value, step)

    def save_weights(self, weights: nn.Module, step: int) -> None:
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)
        fn = os.path.join(self.log_dir, "weights", f"step_{step}.w")
        t.save(
            weights,
            fn
        )

    def _log_scalar_to_file(self, tag: str, val: float, step: int) -> None:
        fn = os.path.join(self.log_dir, f"{tag}.log")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "a") as f:
            f.write(f"{step} {val}\n")

