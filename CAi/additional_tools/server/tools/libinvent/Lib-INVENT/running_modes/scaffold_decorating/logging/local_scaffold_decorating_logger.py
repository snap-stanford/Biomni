import logging

import numpy
from reinvent_chemistry.logging import fraction_valid_smiles
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from running_modes.scaffold_decorating.logging.base_scaffold_decorating_logger import BaseScaffoldDecoratingLogger


class LocalScaffoldDecoratingLogger(BaseScaffoldDecoratingLogger):
    def __init__(self, logging_path):
        super().__init__(logging_path)
        self._logging_path = logging_path
        self._summary_writer = SummaryWriter(log_dir=self._logging_path)
        self._logger = self._setup_logger()

    def log_message(self, message: str):
        self._logger.info(message)

    def log_invalid_smiles(self, message: str):
        self._logger.info(message)

    def log_timestep(self, smiles: str, likelihoods: numpy.ndarray):
        valid_smiles_fraction = fraction_valid_smiles(smiles)
        fraction_unique_entries = self._get_unique_entries_fraction(likelihoods)

        self._summary_writer.add_scalar("Valid_smiles", valid_smiles_fraction)
        self._summary_writer.add_histogram("NLLS", likelihoods)
        self._summary_writer.add_scalar("Unique_smiles", fraction_unique_entries)
        self._summary_writer.add_scalar("Number_of_sampled_compounds", len(smiles))

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("scaffold_decorating_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def __del__(self):
        logging.shutdown()
