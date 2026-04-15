import gzip
import logging
import sys

import tqdm
from reinvent_chemistry.file_reader import FileReader

from running_modes.configurations.tuples_likelihood_computation_configuration import \
    TuplesLikelihoodComputationConfiguration
from running_modes.enums.generative_model_regime import GenerativeModelRegimeEnum
import models.actions as ma
import models.model as mm


class ComputeScaffoldDecorationLikelihoods:
    def __init__(self, configuration: TuplesLikelihoodComputationConfiguration):
        self._configuration = configuration
        self._reader = FileReader([], None)
        self._log = self._get_logger("calculate_nlls")
        self._mode = GenerativeModelRegimeEnum()

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
        model = mm.DecoratorModel.load_from_file(self._configuration.model_path, mode=self._mode.INFERENCE)

        input_csv = self.open_file(self._configuration.input_csv_path, mode="rt")
        if self._configuration.use_gzip:
            self._configuration.output_csv_path += ".gz"
        output_csv = self.open_file(self._configuration.output_csv_path, mode="wt+")

        calc_nlls_action = ma.CalculateNLLsFromModel(model, batch_size=self._configuration.batch_size, logger=self._log)
        scaffold_decoration_list = [fields for fields in self._reader.read_library_design_data_file(self._configuration.input_csv_path, num_fields=2)]

        for nll in tqdm.tqdm(calc_nlls_action.run(scaffold_decoration_list), total=len(scaffold_decoration_list)):
            input_line = input_csv.readline().strip()
            output_csv.write("{}\t{:.8f}\n".format(input_line, nll))

        input_csv.close()
        output_csv.close()

    def open_file(self, path, mode="r", with_gzip=False):
        """
        Opens a file depending on whether it has or not gzip.
        :param path: Path where the file is located.
        :param mode: Mode to open the file.
        :param with_gzip: Open as a gzip file anyway.
        """
        open_func = open
        if path.endswith(".gz") or with_gzip:
            open_func = gzip.open
        return open_func(path, mode)