import logging
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from reinvent_chemistry.logging import fraction_valid_smiles, padding_with_invalid_smiles, \
    check_for_invalid_mols_and_create_legend, find_matching_pattern_in_smiles, add_mols

from diversity_filters.base_diversity_filter import BaseDiversityFilter
from running_modes.configurations.log_configuration import LogConfiguration
from running_modes.reinforcement_learning.logging import BaseReinforcementLogger
from running_modes.reinforcement_learning.logging.console_message import ConsoleMessage
from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class LocalReinforcementLogger(BaseReinforcementLogger):
    def __init__(self, configuration: LogConfiguration):
        super().__init__(configuration)
        self._configuration = configuration
        self._summary_writer = SummaryWriter(log_dir=self._configuration.logging_path)
        # _rows and _columns define the shape of the output grid of molecule images in tensorboard.
        self._rows = 4
        self._columns = 4
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._logger = self._setup_logger()
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter):
        message = self._console_message_formatter.create(start_time, n_steps, step, score_summary,
                                                         agent_likelihood, prior_likelihood,
                                                         augmented_likelihood)
        self._logger.info(message)
        self._tensorboard_report(step, score_summary, agent_likelihood, prior_likelihood, augmented_likelihood,
                                 diversity_filter)

    def _tensorboard_report(self, step, score_summary: FinalSummary, agent_likelihood, prior_likelihood,
                            augmented_likelihood, diversity_filter: BaseDiversityFilter):
        self._summary_writer.add_scalars("nll/avg", {
            "prior": prior_likelihood.mean(),
            "augmented": augmented_likelihood.mean(),
            "agent": agent_likelihood.mean()
        }, step)
        self._summary_writer.add_scalars("nll/variance", {
            "prior": prior_likelihood.var(),
            "augmented": augmented_likelihood.var(),
            "agent": agent_likelihood.var()
        }, step)
        mean_score = np.mean(score_summary.total_score)
        for i, log in enumerate(score_summary.profile):
            self._summary_writer.add_scalar(score_summary.profile[i].name, np.mean(score_summary.profile[i].score),
                                            step)
        self._summary_writer.add_scalar("Valid SMILES", fraction_valid_smiles(score_summary.scored_smiles), step)
        self._summary_writer.add_scalar("Number of SMILES found", diversity_filter.number_of_smiles_in_memory(), step)
        self._summary_writer.add_scalar("Average score", mean_score, step)
        if step % 1 == 0:
            self._log_out_smiles_sample(score_summary.scored_smiles, score_summary.total_score, step, score_summary)

    def _log_out_smiles_sample(self, smiles, score, step, score_summary: FinalSummary):
        self._visualize_structures(smiles, score, step, score_summary)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):

        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        try:
            add_mols(self._summary_writer, "Molecules from epoch", list_of_mols[:self._sample_size], self._rows,
                     [x for x in legends], global_step=step, size_per_mol=(320, 320), pattern=pattern)
        except:
            raise Exception(f"Error in RDKit has occurred, skipping report for step {step}.")

    def _check_for_invalid_mols_and_create_legends(self, smiles, score, score_summary: FinalSummary):
        smiles = padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)

        return list_of_mols, legend, pattern

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.smiles
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("reinforcement_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
