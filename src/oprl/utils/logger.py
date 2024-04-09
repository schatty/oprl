import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


def copy_exp_dir(log_dir: str) -> None:
    cur_dir = os.path.join(os.getcwd(), "src")
    dest_dir = os.path.join(log_dir, "src")
    shutil.copytree(cur_dir, dest_dir)
    logging.info(f"Source copied into {dest_dir}")


def save_json_config(config: dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(config, f)


class Logger(ABC):

    def log_scalars(self, values: dict[str, float], step: int):
        """
        Args:
            values: Dict with tag -> value to log.
            step: Iter step.
        """
        (self.log_scalar(k, v, step) for k, v in values.items())

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        logging.info(f"{tag}\t{value}\tat step {step}")

    @abstractmethod
    def log_video(self, tag: str, imgs, step: int):
        logging.warning("Skipping logging video in STDOUT logger")


class StdLogger(Logger):
    def __init__(self, *args, **kwargs):
        pass

    def log_scalar(self, tag: str, value: float, step: int):
        logging.info(f"{tag}\t{value}\tat step {step}")

    def log_video(self, *args, **kwargs):
        logging.warning("Skipping logging video in STDOUT logger")


class FileLogger(Logger):
    def __init__(self, logdir: str, config: dict[str, Any]):
        self.writer = SummaryWriter(logdir)

        self._log_dir = logdir

        logging.info(f"Source code is copied to {logdir}")
        copy_exp_dir(logdir)
        save_json_config(config, os.path.join(logdir, "config.json"))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        self._log_scalar_to_file(tag, value, step)

    def log_video(self, tag: str, imgs, step: int) -> None:
        os.makedirs(os.path.join(self._log_dir, "images"))
        fn = os.path.join(self._log_dir, "images", f"{tag}_step_{step}.npz")
        with open(fn, "wb") as f:
            np.save(f, imgs)

    def _log_scalar_to_file(self, tag: str, val: float, step: int) -> None:
        fn = os.path.join(self._log_dir, f"{tag}.log")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "a") as f:
            f.write(f"{step} {val}\n")
