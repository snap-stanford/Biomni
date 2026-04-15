from typing import List, Dict

import numpy as np
from reinvent_chemistry.library_design import FragmentReactions

from reaction_filters.base_reaction_filter import BaseReactionFilter
from running_modes.configurations.reaction_filter_configuration import ReactionFilterConfiguration


class SelectiveFilter(BaseReactionFilter):
    def __init__(self, configuration: ReactionFilterConfiguration):
        self._chemistry = FragmentReactions()
        self._reactions = self._configure_reactions(configuration.reactions)
        self._score_cutoff = 0.5

    def _configure_reactions(self, reaction_smarts: Dict[str, List[str]]):
        reactions = {}
        for key, smarts_list in reaction_smarts.items():
            converted = self._chemistry.create_reactions_from_smarts(smarts_list)
            reactions[int(key)] = converted
        return reactions

    def evaluate(self, molecule):
        if not self._reactions:
            return 1
        return self.score_molecule(molecule)

    def score_molecule(self, molecule):
        new_bonds = self._find_new_bonds(molecule)
        count = self._count_applicable_reactions_on_molecule(molecule, new_bonds)
        score = self._score_cutoff + (1 - self._score_cutoff)*(count / len(new_bonds))
        return score

    def _find_new_bonds(self, molecule) -> dict:
        """Find atoms marked with the bondNum atom property and add to a list"""
        _bond_indices_dict = {}
        for atom in molecule.GetAtoms():
            if atom.HasProp("bondNum"):
                bondNum = int(atom.GetProp("bondNum"))
                if not bondNum in _bond_indices_dict:
                    _bond_indices_dict[bondNum] = []
                _bond_indices_dict[bondNum].append(atom.GetIdx())
        return _bond_indices_dict

    def _convert_reactants_to_atom_indices(self, reactant_pairs):
        """Convert the list of tuples of reactants into a list of lists of original atom indices"""
        _reactant_idx_list = []
        for reactant_pair in reactant_pairs:
            outcome_list = []
            for reactant in reactant_pair:
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

    def _count_applicable_reactions_on_molecule(self, molecule, target_bonds: dict):
        count = 0
        for bond_indx, reactions in self._reactions.items():
            if reactions:
                reaction_pairs = self._apply_reactions_on_bond(molecule, reactions)
                reactant_idxs = self._convert_reactants_to_atom_indices(reaction_pairs)
                if self._detect_sliced_bond_by_reaction(target_bonds[bond_indx], reactant_idxs):
                    count += 1
            else:
                count += 1
        return count

    def _apply_reactions_on_bond(self, molecule, reactions: List) -> List:
        outcomes = []
        for reaction in reactions:
            outcome = reaction.RunReactant(molecule, 0)
            outcomes.extend(outcome)
        reaction_pairs = [outcome for outcome in outcomes]
        return reaction_pairs

    def _detect_sliced_bond_by_reaction(self, bond, reactant_idxs):
        """Test a given bond if its targetable by any retrosynthethic disconnection"""
        return np.any(
            [
                self._verify_atom_splits(bond, sets[0], sets[1])
                for sets in reactant_idxs
                if len(sets) == 2
            ]
        )

    def _verify_atom_splits(self, bond, set1, set2) -> bool:
        """Test if the bond is in split into the two different sets"""
        atom1_set_id = self._find_set_id(bond[0], set1, set2)
        atom2_set_id = self._find_set_id(bond[1], set1, set2)
        atoms_are_from_different_sets = (atom1_set_id != atom2_set_id) and (atom1_set_id != 0) and (atom2_set_id != 0)
        return atoms_are_from_different_sets

    def _find_set_id(self, idx, set1, set2) -> int:
        """Check if an idx is in either of two sets or none of them"""
        if idx in set1:
            return 1
        elif idx in set2:
            return 2
        else:
            return 0
