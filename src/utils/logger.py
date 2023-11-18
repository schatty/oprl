import os

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir: str):
        self.writer = SummaryWriter(logdir)

        self._log_dir = logdir
        self._tags_to_log_file = ("reward", )

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

        for tag_keyword in self._tags_to_log_file:
            if tag_keyword in tag:
                self._log_scalar_to_file(tag, value, step)

    def _log_scalar_to_file(self, tag: str, val: float, step: int):
        fn = os.path.join(self._log_dir, f"{tag}.log")
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, "a") as f:
            f.write(f"{step} {val}\n")

