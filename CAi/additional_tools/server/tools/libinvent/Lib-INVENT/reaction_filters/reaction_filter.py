from reaction_filters import ReactionFiltersEnum
from reaction_filters.base_reaction_filter import BaseReactionFilter
from reaction_filters.non_selective_filter import NonSelectiveFilter
from reaction_filters.selective_filter import SelectiveFilter
from running_modes.configurations.reaction_filter_configuration import ReactionFilterConfiguration


class ReactionFilter:
    def __new__(cls, configuration: ReactionFilterConfiguration) -> BaseReactionFilter:
        enum = ReactionFiltersEnum()
        if configuration.type == enum.NON_SELECTIVE:
            return NonSelectiveFilter(configuration)
        elif configuration.type == enum.SELECTIVE:
            return SelectiveFilter(configuration)
        else:
            raise TypeError(f"Requested filter type: '{configuration.type}' is not implemented.")
