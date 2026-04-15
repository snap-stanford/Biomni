import json
import logging
import os
from abc import ABC, abstractmethod

import pandas as pd
from typing import List

from running_modes.configurations import ConfigurationEnvelope, ScoringConfiguration
from running_modes.enums.scoring_runner_enum import ScoringRunnerEnum
from reinvent_scoring.scoring.score_summary import FinalSummary


class BaseScoringLogger(ABC):
    def __init__(self, configuration: ConfigurationEnvelope):
        self._configuration = configuration
        self._log_config = ScoringConfiguration(**self._configuration.parameters)
        self._setup_workfolder()
        self._logger = self._setup_logger()
        self._scoring_runner_enum = ScoringRunnerEnum()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.output_folder, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_results(self, score_summary: FinalSummary):
        output_file = os.path.join(self._log_config.output_folder, "scored_smiles.csv")
        table_header = self._create_table_header(score_summary)
        data_list = self._convolute_score_summary(score_summary)
        dataframe = pd.DataFrame(data_list, columns=table_header, dtype=str)
        dataframe.to_csv(output_file, header=True, index=False)

    def _convolute_score_summary(self, score_summary: FinalSummary) -> []:
        smiles = score_summary.scored_smiles
        scores = score_summary.total_score
        component_scores = [c.score for c in score_summary.profile]
        data = []

        for indx in range(len(smiles)):
            valid = 1 if indx in score_summary.valid_idxs else 0
            score = scores[indx] if valid else 0
            row = self._compose_row_entry(indx, valid, score, smiles[indx], component_scores)
            data.append(row)

        return data

    def _compose_row_entry(self, indx: int, valid: int, score: float, smile: str, component_scores: List) -> List:
        row = [smile, score]
        components = [component[indx] for component in component_scores]
        row.extend(components)
        row.append(valid)
        return row

    def _create_table_header(self, score_summary: FinalSummary) -> List:
        column_names = [self._scoring_runner_enum.SMILES, self._scoring_runner_enum.TOTAL_SCORE]
        component_names = [c.name for c in score_summary.profile]
        column_names.extend(component_names)
        column_names.append(self._scoring_runner_enum.VALID)

        return column_names

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("scoring_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.output_folder):
            os.makedirs(self._log_config.output_folder)
