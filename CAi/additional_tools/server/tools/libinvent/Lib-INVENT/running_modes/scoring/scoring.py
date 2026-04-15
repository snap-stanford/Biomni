from reinvent_chemistry import Standardizer, Conversions
from reinvent_scoring import ScoringFunctionFactory
from reinvent_scoring.scoring.score_summary import FinalSummary

from reaction_filters.reaction_filter import ReactionFilter
from running_modes.configurations import ConfigurationEnvelope, ScoringConfiguration
from running_modes.scoring.logging.scoring_logger import ScoringLogger


class Scoring:
    def __init__(self, configuration: ConfigurationEnvelope, config: ScoringConfiguration):
        self._scoring_function = ScoringFunctionFactory(config.scoring_function)
        self._config = config
        self._reaction_filter = ReactionFilter(config.reaction_filter)
        self._logger = ScoringLogger(configuration)
        self._standardization = Standardizer()
        self._conversion = Conversions()

    def run(self):
        input_smiles = list(self._standardization.read_smiles_file(file_path=self._config.input, randomize=False, standardize=False))
        final_score: FinalSummary = self._scoring_function.get_final_score(input_smiles)

        self._logger.log_results(score_summary=final_score)
        self._logger.log_out_input_configuration()
