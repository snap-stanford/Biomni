

class GenerativeModelParametersEnum:
    NUMBER_OF_LAYERS = "num_layers"
    NUMBER_OF_DIMENSIONS = "num_dimensions"
    VOCABULARY_SIZE = "vocabulary_size"
    DROPOUT = "dropout"

    # try to find the internal value and return
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    # prohibit any attempt to set any values
    def __setattr__(self, key, value):
        raise ValueError("No changes allowed.")