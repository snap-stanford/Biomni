import json
import os

from dacite import from_dict

from models.model import DecoratorModel
from running_modes.configurations import ReinforcementLearningConfiguration, ScaffoldDecoratingConfiguration, \
    TransferLearningConfiguration, ConfigurationEnvelope, ScoringConfiguration
from running_modes.configurations.create_model_configuration import CreateModelConfiguration
from running_modes.configurations.tuples_likelihood_computation_configuration import \
    TuplesLikelihoodComputationConfiguration
from running_modes.create_model.create_model import CreateModel
from running_modes.enums import GenerativeModelRegimeEnum, RunningModeEnum
from running_modes.reinforcement_learning.logging import ReinforcementLogger
from running_modes.reinforcement_learning.reinforcement_learning import ReinforcementLearning
from running_modes.scaffold_decorating.logging.scaffold_decorating_logger import ScaffoldDecoratingLogger
from running_modes.scaffold_decorating.scaffold_decoration import ScaffoldDecorator
from running_modes.scoring.scoring import Scoring
from running_modes.transfer_learning.transfer_learning import LargeScaleTransferLearning
from running_modes.tuples_likelihood_computation.tuples_likelihood_computation import \
    ComputeScaffoldDecorationLikelihoods


class Manager:

    def __init__(self, configuration):
        self._configuration = from_dict(data_class=ConfigurationEnvelope, data=configuration)
        self._model_regime = GenerativeModelRegimeEnum()
        self._load_environmental_variables()

    def _scaffold_decorating(self):
        config = from_dict(data_class=ScaffoldDecoratingConfiguration, data=self._configuration.parameters)
        logger = ScaffoldDecoratingLogger(config.logging_path)
        scaffold_decorator = ScaffoldDecorator(config, logger)
        scaffold_decorator.run()

    def _transfer_learning(self):
        config = from_dict(data_class=TransferLearningConfiguration, data=self._configuration.parameters)

        transfer_learning = LargeScaleTransferLearning(config)
        transfer_learning.run()

    def _reinforcement_learning(self):
        config = from_dict(data_class=ReinforcementLearningConfiguration, data=self._configuration.parameters)

        critic = DecoratorModel.load_from_file(config.critic, mode=self._model_regime.INFERENCE)
        actor = DecoratorModel.load_from_file(config.actor, mode=self._model_regime.TRAINING)
        logger = ReinforcementLogger(self._configuration.logging)

        reinforcement_learning = ReinforcementLearning(critic=critic, actor=actor, configuration=config, logger=logger)
        reinforcement_learning.run()

    def _scoring(self):
        scoring_config = from_dict(data_class=ScoringConfiguration, data=self._configuration.parameters)
        scoring_mode = Scoring(self._configuration, scoring_config)
        scoring_mode.run()

    def _create_model(self):
        create_model_config = from_dict(data_class=CreateModelConfiguration, data=self._configuration.parameters)
        model_creator = CreateModel(create_model_config)
        model_creator.run()

    def _compute_tuples_likelihoods(self):
        nlls_config = from_dict(data_class=TuplesLikelihoodComputationConfiguration,
                                data=self._configuration.parameters)
        nlls_calculator = ComputeScaffoldDecorationLikelihoods(nlls_config)
        nlls_calculator.run()

    def run(self):
        """determines from the configuration object which type of run it is expected to start"""
        running_mode = RunningModeEnum()
        switcher = {
            running_mode.SCAFFOLD_DECORATING: self._scaffold_decorating,
            running_mode.TRANSFER_LEARNING: self._transfer_learning,
            running_mode.REINFORCEMENT_LEARNING: self._reinforcement_learning,
            running_mode.SCORING: self._scoring,
            running_mode.CREATE_MODEL: self._create_model,
            running_mode.TUPLES_LIKELIHOOD_COMPUTATION: self._compute_tuples_likelihoods
        }
        job = switcher.get(self._configuration.run_type, lambda: TypeError)
        job()

    def _load_environmental_variables(self):
        try:
            project_root = os.path.dirname(__file__)
            with open(os.path.join(project_root, '../configurations/config.json'), 'r') as f:
                config = json.load(f)
            environmental_variables = config["ENVIRONMENTAL_VARIABLES"]
            for key, value in environmental_variables.items():
                os.environ[key] = value

        except KeyError as ex:
            raise ex
