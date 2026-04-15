import glob
import itertools as it
import os.path

import torch
from reinvent_chemistry.file_reader import FileReader

import models.model as mm
from models.actions import TrainModel
from running_modes.configurations.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.transfer_learning.logging.transfer_learning_logger import TransferLearningLogger


class LargeScaleTransferLearning:
    def __init__(self, configuration: TransferLearningConfiguration):
        self._reader = FileReader([], None)
        self.config = configuration
        self.model = mm.DecoratorModel.load_from_file(self.config.model_path)
        self._set_up_output_folder()

        self._logger = TransferLearningLogger(self.config.logging_path, self.config.with_weights)

        self.training_sets = self.load_sets(self.config.training_set_path)
        self.validation_sets = self.load_sets(self.config.validation_sets_path)

    def load_sets(self, set_path):
        file_paths = [set_path]
        if os.path.isdir(set_path):
            file_paths = sorted(glob.glob(f"{set_path}/*.smi"))

        for path in it.cycle(file_paths):  # stores the path instead of the set
            yield list(self._reader.read_library_design_data_file(path, num_fields=2))

    def run(self):
        # setup
        optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self.config.learning_rate.start)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=self.config.learning_rate.step,
                                                       gamma=self.config.learning_rate.gamma)

        # train
        TrainModel(model=self.model, optimizer=optimizer, training_sets=self.training_sets,
                   validation_sets=self.validation_sets, logger=self._logger, configuration=self.config,
                   lr_scheduler=lr_scheduler).run()

    def _set_up_output_folder(self):
        if not os.path.isdir(self.config.output_path):
            os.makedirs(self.config.output_path)
