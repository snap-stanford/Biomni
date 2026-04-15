**Please note: this repository is no longer being maintained.**

Implementation of the Lib-INVENT decorator model
=======================================================================================================================================

This repository holds the code used in the publication [Lib-INVENT: Reaction Based Generative Scaffold Decoration for in silico Library Design](https://chemrxiv.org/articles/preprint/Lib-INVENT_Reaction_Based_Generative_Scaffold_Decoration_for_in_silico_Library_Design/14473980). It holds the code needed to use the tool for chemical library generation as well as the initial training of the generative model.
The code for preprocessing of the dataset is contained in a separate project called Library Design Datasets. 

The scripts are organised in the following folder structure:

(1) running modes: A folder containing the core python scripts. These include the key functionalities as well as additional inference modes.
- transfer learning: The teacher forcing used for pretraining of the prior.
- reinforcement learning: The RL loop used for focused library generation. This includes the four different learning strategies as described in the manuscriped. The DAP strategy is currently recommended for use in practical applications.
- scaffold decorating: A sampling mode proposing decorations for a list of input scaffolds, given a trained decorator.
- scoring: Assigns scores to input scaffolds based on a user-specified scoring function.
- tuples likelihood computation: Calculates the likelihoods of scaffold-decoration pairs according to a given model. <br>

(2) training_sets: The [datasets](https://doi.org/10.5281/zenodo.6627127) used for training of the prior.
- purged_chembl_sliced.smi.gz: The CHEMBl 27 compounds, filtered according to the rules described in the manuscript and sliced according to the reaction rules.
- chembl_train.smi.gz: The purged, sliced dataset used for model training. The DRD2 compounds are removed as described in the manuscript.

(3) trained_models: The two models used for experiments:
- reaction_based.model: The pretrained prior based on the dataset processed using reaction based slicing.
- recap.model: The prior trained on CHEMBl slices according to RECAP rules.

(4) examples: example JSON configuration files for the running modes.

(5) configurations: Setup of local paths and licenses. The paths within the JSON file are to be updated as appropriate for the user.

(6) diversity_filters and reaction_filters: The various filters used in training of the reinforcement learning model.

(7) the main folder: The entry points for all the running modes and unit tests.

(8) tutorial: Jupyter notebooks containing demos for setting up the input JSON configuration files for the various running modes.

(9) reaction definitions: The reaction SMIRKS used to slice the CHEMBl dataset.

Requirements
------------
The repository includes a Conda `environment.yml` file with the required libraries to run all the scripts. All models were tested on Linux with both a Tesla V-100 and a Geforce 2070. It should work just fine with other Linux setups and a mid-high range GPU.

Install
-------
A [Conda](https://conda.io/miniconda.html) `environment.yml` is supplied with all the required libraries.

~~~~
$> git clone <repo url>
$> cd <repo folder>
$> conda env create -f environment.yml
$> conda activate lib-invent
(lib-invent) $> ...
~~~~
NOTE: Sometimes issues with the installation of tensorflow appear. It may need to be reinstalled manually.
From here the general usage applies.

General Usage
-------------
The functionality of the model is separated into distinct mutually independent running modes. 
They are called through the `input.py` file and corresponding JSON configuration files specifying the parameters necessary to execute the given running mode. An example configuration is provided for each running mode in the `examples` folder.
To run, type:

`python <path/to/input.py> <path/to/configuration.json>`

All output files are in tsv format (the separator is \t). This is also the expected format of the input files where these correspond to a dataset.

#### Configurations
Each configuration JSON file should contain a field `"run_type"`, which determines the running mode to be invoked. The strings to be passed as an argument can be found contained in the `running_modes/enums/running_mode_enum.py` file.
The second field is called `"parameters"` and takes a dictionary as an input. This dictionary contains all the arguments specified by the configuration of the appropriate running mode - these are defined in the `running_modes/configurations` folder.
In this way, any Lib-INVENT functionality can be run from the command line with a single command containing only two arguments: the path to the input.py file and to the JSON configuration.

#### Overview
1) Create model: <br>
The first running mode to run. It takes a training dataset and initiates an empty model with the appropriate vocabulary and architecture specified by the configuration file. This model is then typically passed to the transfer learning mode to be trained to produce valid smiles. 

2) Transfer learning: <br>
This mode corresponds to training a decorator model to maximise the likelihood of scaffold-decoration pairs produced by it. The decorator can be initiated as an empty model (in which case the training essentially amounts to teacher's forcing) or a pretrained decorator (the transfer learning case). In either case, the vocabulary has to correspond to the vocabulary present in the trained and validation datasets.

3) Reinforcement learning: <br>
This mode takes a pretrained decorator model and attempts to adapt it to a specific task using reinforcement learning. Essentially, policy iterations are performed and the reward contains scores computed by the scoring function.

4) Scaffold decorating (called sampling in previous versions of the project): <br>
As opposed to the previous running modes, no model is trained here. Instead, this running produces decorations for scaffolds passed to an already trained model. It is useful for evaluation purposes and for practical usage of the models.

5) Scoring: <br>
A running mode used to compute the scores given by a scoring function for a set of input smiles. It is used to evaluate the utility of compounds proposed by the generative models.

6) Tuples likelihood computation: <br>
Computes the likelihood of scaffold-decoration pairs passed to a trained model. This is useful for evaluation purposes.



Usage examples
--------------
To download the [datasets](https://doi.org/10.5281/zenodo.6627127) from [Zenodo](https://zenodo.org/): `python download_training_datasets.py`

To run with a single line command from the command line (provided the paths in the configuration JSON file are adapted as required):
`python input.py examples/scoring.json`

Testing:
`python main_test.py`



