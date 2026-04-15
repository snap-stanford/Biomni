from typing import List

from reinvent_chemistry.logging import fraction_valid_smiles, add_mols
from torch.utils.tensorboard import SummaryWriter

from running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger


class LocalTransferLearningLogger(BaseTransferLearningLogger):
    """Collects stats for an existing RNN model."""

    def __init__(self, logging_path: str, weights: bool=False):
        super().__init__(logging_path)
        self._summary_writer = SummaryWriter(log_dir=self._logging_path)
        self._with_weights = weights

    def log_timestep(self, lr, epoch, training_smiles, validation_smiles,
                     validation_nlls, training_nlls, jsd_data_no_bins, jsd_data_bins, model):
        self.log_message(f"Collecting data for epoch {epoch}")

        if self._with_weights:
            self._weight_stats(model, epoch)

        self._summary_writer.add_histogram("nll_plot/training", training_nlls, epoch)
        self._summary_writer.add_scalar("nll/mean_training", training_nlls.mean(), epoch)
        self._summary_writer.add_scalar(("nll_variance/training"), training_nlls.var(), epoch)

        # TODO Maybe have option of no validation data?
        self._summary_writer.add_histogram("nll_plot/validation", validation_nlls, epoch)
        self._summary_writer.add_scalar("nll/mean_validation", validation_nlls.mean(), epoch)
        self._summary_writer.add_scalar("nll_variance/validation", validation_nlls.var(), epoch)

        self._valid_stats(training_smiles, "train", epoch)
        self._valid_stats(validation_smiles, "validation", epoch)
        self._visualize_structures(training_smiles, epoch)
        self._summary_writer.add_scalar('jsd_binned', jsd_data_bins, epoch)
        self._summary_writer.add_scalar('jsd_no_bin', jsd_data_no_bins, epoch)
        self._summary_writer.add_scalar("lr", lr, epoch)

    def _valid_stats(self, smiles, name, epoch):
        self._summary_writer.add_scalar(f"valid/{name}", fraction_valid_smiles(smiles), epoch)

    def _weight_stats(self, model, epoch):
        for name, weights in model.network.named_parameters():
            self._summary_writer.add_histogram(f"weights/{name}", weights.clone().cpu().data.numpy(), epoch)

    def _visualize_structures(self, smiles: List[str], epoch: int):
        list_of_labels, list_of_mols = self._count_compound_frequency(smiles)
        if len(list_of_mols) > 0:
            add_mols(self._summary_writer, "Most Frequent Molecules", list_of_mols, self._rows, list_of_labels,
                     global_step=epoch)