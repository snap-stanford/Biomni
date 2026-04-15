
class RunningModeEnum:
    TRANSFER_LEARNING = "transfer_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    SCAFFOLD_DECORATING = "scaffold_decorating"
    CREATE_MODEL = "create_model"
    SCORING = "scoring"
    TUPLES_LIKELIHOOD_COMPUTATION = "compute_tuples_likelihoods"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")
