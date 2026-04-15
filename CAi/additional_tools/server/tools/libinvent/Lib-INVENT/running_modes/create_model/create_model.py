import logging
import sys

from reinvent_chemistry.file_reader import FileReader

import models.decorator as md
import models.model as mm
import models.vocabulary as mv
from running_modes.configurations.create_model_configuration import CreateModelConfiguration
from running_modes.enums import GenerativeModelParametersEnum


class CreateModel:
    def __init__(self, configuration: CreateModelConfiguration):
        self._configuration = configuration
        self._reader = FileReader([], None)
        self._log = self._get_logger(name="create_model")

    def _get_logger(self, name, level=logging.INFO):

        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    def run(self):
        parameter_enum = GenerativeModelParametersEnum()
        scaffold_list, decoration_list = zip(*self._reader.read_library_design_data_file(self._configuration.input_smiles_path, num_fields=2))

        self._log.info("Building vocabulary")

        vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)

        self._log.info("Scaffold vocabulary contains %d tokens: %s",
                 vocabulary.len_scaffold(), vocabulary.scaffold_vocabulary.tokens())
        self._log.info("Decorator vocabulary contains %d tokens: %s",
                 vocabulary.len_decoration(), vocabulary.decoration_vocabulary.tokens())

        encoder_params = {
            parameter_enum.NUMBER_OF_LAYERS: self._configuration.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self._configuration.layer_size,
            parameter_enum.VOCABULARY_SIZE: vocabulary.len_scaffold(),
            parameter_enum.DROPOUT: self._configuration.dropout
        }

        decoder_params = {
            parameter_enum.NUMBER_OF_LAYERS: self._configuration.num_layers,
            parameter_enum.NUMBER_OF_DIMENSIONS: self._configuration.layer_size,
            parameter_enum.VOCABULARY_SIZE: vocabulary.len_decoration(),
            parameter_enum.DROPOUT: self._configuration.dropout
        }

        decorator = md.Decorator(encoder_params, decoder_params)
        model = mm.DecoratorModel(vocabulary, decorator, self._configuration.max_sequence_length)

        self._log.info("Saving model at %s", self._configuration.output_model_path)
        model.save(self._configuration.output_model_path)
