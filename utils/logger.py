import os
import torch
import pickle
from torchnet.logger import MeterLogger
from torchnet.utils import ResultsWriter
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Logger(object):

    def __init__(self, log_path, server='localhost', env='main', port=8097):
        """
        General logger.

        Args:
            log_dir (str): log directory
            server (str): server ip address
            env (str): visdom environment name
            port (int): visdom port
            title (str): visdom plot prefix
        """
        self.writer = ResultsWriter(log_path, overwrite=True)
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning
        self.log_name = log_path[log_path.rfind('/')+1:]
        self.scalar_meter = MeterLogger(server=server, env=env, port=port)

    def scalar_summary(self, tag, value):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
        """
        plot_name = f"{self.log_name}-{tag}"
        self.scalar_meter.update_loss(torch.from_numpy(np.asarray(value)).unsqueeze(0), meter=plot_name)
        self.writer.update(tag, value)

