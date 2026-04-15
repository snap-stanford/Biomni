import os
from abc import ABC, abstractmethod

import torch
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum

from reinvent_scoring.scoring.score_summary import FinalSummary

from running_modes.configurations.log_configuration import LogConfiguration


class BaseReinforcementLogger(ABC):
    def __init__(self, logging_path: LogConfiguration):
        self._configuration = logging_path

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter):
        raise NotImplementedError("timestep_report method is not implemented")

    def save_filter_memory(self, diversity_filter):
        diversity_memory = diversity_filter.get_memory_as_dataframe()
        # TODO: Pass a job_name parameter from the config
        self.save_to_csv(diversity_memory, self._configuration.logging_path)

    def save_to_csv(self, scaffold_memory, path, job_name="default_job"):

        sf_enum = ScoringFunctionComponentNameEnum()
        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, "scaffold_memory.csv")

        if len(scaffold_memory) > 0:
            sorted_df = scaffold_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            # sorted_df["job_name"] = [f"{job_name}" for _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file_name, index=False)
