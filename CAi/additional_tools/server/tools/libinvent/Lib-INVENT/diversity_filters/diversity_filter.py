from diversity_filters import NoFilter, NoFilterWithPenalty
from diversity_filters.base_diversity_filter import BaseDiversityFilter
from diversity_filters.diversity_filter_parameters import DiversityFilterParameters


class DiversityFilter:

    def __new__(cls, parameters: DiversityFilterParameters) -> BaseDiversityFilter:
        all_filters = dict(NoFilter=NoFilter,
                           NoFilterWithPenalty=NoFilterWithPenalty)
        div_filter = all_filters.get(parameters.name)
        return div_filter(parameters)
