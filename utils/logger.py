from torchnet.logger import MeterLogger
from torchnet.utils import ResultsWriter
import logging

logger = logging.getLogger("main")


class Logger(object):

    def __init__(self, log_path):
        """
        General logger.

        Args:
            log_dir (str): log directory
        """

        self.writer = ResultsWriter(log_path, overwrite=True)
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning
        self.log_name = log_path[log_path.rfind('/')+1:]

    def scalar_summary(self, tag, value):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
        """
        self.writer.update(tag, value)
