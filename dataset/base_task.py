class base_task:
    def __init__(self):
        pass

    def get_example(self):
        pass

    def get_iterator(self):
        pass

    def evaluate(self):
        pass

    def output_class(self):
        pass

    def get_prompt_from_input(self, input):
        return self.get_example(input)["prompt"]
