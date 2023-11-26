import json
import shutil
import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def copy_exp_dir(log_dir: str):
    cur_dir = os.path.join(os.getcwd(), "src")
    dest_dir = os.path.join(log_dir, "src")
    shutil.copytree(cur_dir, dest_dir)
    print(f"Source copied into {dest_dir}")


def save_json_config(config: str, path: str):
    with open(path, "w") as f:
        json.dump(config, f)


class Logger:
    def __init__(self, logdir: str, config: str):
        self.writer = SummaryWriter(logdir)

        self._log_dir = logdir
        self._tags_to_log_file = ("reward", )

        print("Will copy file to ", logdir)
        copy_exp_dir(logdir)
        save_json_config(config, os.path.join(logdir, "config.json"))


    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

        for tag_keyword in self._tags_to_log_file:
            if tag_keyword in tag:
                self._log_scalar_to_file(tag, value, step)

    def log_video(self, tag: str, imgs, step: int):
        # TODO: Log to TensorBoard
        os.makedirs(os.path.join(self._log_dir, "images"))
        fn = os.path.join(self._log_dir, "images", f"{tag}_step_{step}.npz")
        with open(fn, "wb") as f:
            np.save(f, imgs)
        print("Video logged!")

    def _log_scalar_to_file(self, tag: str, val: float, step: int):
        fn = os.path.join(self._log_dir, f"{tag}.log")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "a") as f:
            f.write(f"{step} {val}\n")

