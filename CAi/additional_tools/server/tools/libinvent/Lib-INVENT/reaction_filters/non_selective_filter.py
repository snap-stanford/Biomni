from typing import Dict, List

import numpy as np
from reinvent_chemistry.library_design import FragmentReactions

from reaction_filters.base_reaction_filter import BaseReactionFilter
from running_modes.configurations.reaction_filter_configuration import ReactionFilterConfiguration


class NonSelectiveFilter(BaseReactionFilter):
    def __init__(self, configuration: ReactionFilterConfiguration):
        self._chemistry = FragmentReactions()
        self._reactions = self._configure_reactions(configuration.reactions)

    def _configure_reactions(self, reaction_smarts: Dict[str, List[str]]):
        all_reactions = []
        for smirks in reaction_smarts.values():
            reactions = self._chemistry.create_reactions_from_smarts(smirks)
            all_reactions.extend(reactions)
        return reactions

    def evaluate(self, molecule):
        if not self._reactions:
            return 1
        return self.score_molecule(molecule)

    def score_molecule(self, molecule):
        bond_indices = self._get_created_bonds(molecule)
        synthons = self._run_reactions(molecule)
        reactant_idxs = self._analyze_reactants(synthons)
        score = self._score_mol(bond_indices, reactant_idxs)
        return score

    def _analyze_reactants(self, synthons):
        """Convert the list of tuples of reactants into a list of lists of original atom indices"""
        _reactant_idx_list = []
        for synthon in synthons:
            outcome_list = []
            for reactant in synthon:
                idxs = set(
                    [
                        int(atom.GetProp("react_atom_idx"))
                        for atom in reactant.GetAtoms()
                        if atom.HasProp("react_atom_idx")
                    ]
                )
                outcome_list.append(idxs)
            _reactant_idx_list.append(outcome_list)
        return _reactant_idx_list

    def _get_created_bonds(self, molecule):
        """Find atoms marked with the bondNum atom property and add to a list"""
        _bond_indices_dict = {}
        for atom in molecule.GetAtoms():
            if atom.HasProp("bondNum"):
                bondNum = int(atom.GetProp("bondNum"))
                if not bondNum in _bond_indices_dict:
                    _bond_indices_dict[bondNum] = []
                _bond_indices_dict[bondNum].append(atom.GetIdx())
        return _bond_indices_dict

    def _get_list_num(self, idx, set1, set2):
        """Check if an idx is in either of two sets or none of them"""
        if idx in set1:
            return 1
        elif idx in set2:
            return 2
        else:
            return 0

    def _run_reactions(self, molecule):
        """Build full list of possible splits"""
        synthons = []
        for reaction in self._reactions:
            outcomes = reaction.RunReactant(molecule, 0)
            [synthons.append(outcome) for outcome in outcomes]
        return synthons

    def _score_mol(self, bond_indices_dict, reactant_idxs):
        """Score the current state by checking if all bonds has a possible disconnection"""
        disconnections = [self._test_bond(bond, reactant_idxs) for bond in bond_indices_dict.values()]
        return np.sum(disconnections) / len(disconnections)

    def _test_splitting(self, bond, set1, set2):
        """Test if the bond is in split into the two different sets"""
        atom1_set = self._get_list_num(bond[0], set1, set2)
        atom2_set = self._get_list_num(bond[1], set1, set2)
        return (atom1_set != atom2_set) and (atom1_set != 0) and (atom2_set != 0)

    def _test_bond(self, bond, reactant_idxs):
        """Test a given bond if its targetable by any retrosynthethic disconnection"""
        return np.any(
            [
                self._test_splitting(bond, sets[0], sets[1])
                for sets in reactant_idxs
                if len(sets) == 2
            ]
        )
