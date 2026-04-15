import pandas as pd
from reinvent_chemistry import Conversions
from reinvent_chemistry.enums import FilterTypesEnum
from reinvent_chemistry.file_reader import FileReader
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints
from reinvent_chemistry.standardization.filter_configuration import FilterConfiguration

from models.model import DecoratorModel
from models.rl_actions import SampleModel
from running_modes.configurations import ScaffoldDecoratingConfiguration
from running_modes.enums import GenerativeModelRegimeEnum


class ScaffoldDecorator:
    def __init__(self, configuration: ScaffoldDecoratingConfiguration, logger):
        self._configuration = configuration
        self._logger = logger
        self._model_regime = GenerativeModelRegimeEnum()
        self._bond_maker = BondMaker()
        self._attachment_points = AttachmentPoints()
        self._conversion = Conversions()

        df_dict = {"SMILES": [], "Scaffold": [], "Decorations": [], "Likelihoods": []}
        self._decorated_scaffolds = pd.DataFrame(df_dict)

        filter_types = FilterTypesEnum()
        config = FilterConfiguration(name=filter_types.GET_LARGEST_FRAGMENT, parameters={})
        self._reader = FileReader([config], logger)

    def run(self):
        model = DecoratorModel.load_from_file(self._configuration.model_path, mode=self._model_regime.INFERENCE)
        input_scaffolds = list(
            self._reader.read_delimited_file(self._configuration.input_scaffold_path, standardize=True))

        input_scaffolds = [scaffold for scaffold in input_scaffolds if scaffold]
        input_scaffolds = input_scaffolds * self._configuration.number_of_decorations_per_scaffold

        sampling_action = SampleModel(model, self._configuration.batch_size, self._logger,
                                      self._configuration.randomize, sample_uniquely=self._configuration.sample_uniquely)
        sampled_sequences = sampling_action.run(input_scaffolds)

        for sample in sampled_sequences:
            scaffold = self._attachment_points.add_attachment_point_numbers(sample.scaffold, canonicalize=False)
            molecule = self._bond_maker.join_scaffolds_and_decorations(scaffold, sample.decoration)

            if molecule:
                smile = self._conversion.mol_to_smiles(molecule, isomericSmiles=False, canonical=False)

                series = pd.Series([smile, sample.scaffold, sample.decoration, sample.nll],
                                   index=['SMILES', 'Scaffold', 'Decorations', 'Likelihoods'])

                self._decorated_scaffolds = pd.concat(
                    [self._decorated_scaffolds, series.to_frame().T],
                    ignore_index=True
                )
            else:
                self._logger.log_message(f"Invalid decorations: {sample.decoration} for scaffold {sample.scaffold}")

        self._logger.log_message(f"Sampled {len(self._decorated_scaffolds)} scaffolds in total")
        # self._logger.log_timestep(self._decorated_scaffolds['SMILES'].values,
        #                           self._decorated_scaffolds['Likelihoods'].values,,

        self._decorated_scaffolds.to_csv(self._configuration.output_path, index=False)
