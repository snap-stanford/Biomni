from biomni.llm import get_llm


class base_agent:
    def __init__(self, llm="claude-3-haiku-20240307", cheap_llm=None, tools=None, temperature=0.7):
        self.tools = tools
        self.llm = get_llm(llm, temperature)
        if cheap_llm is None:
            self.cheap_llm = llm
        else:
            self.cheap_llm = cheap_llm

    def configure(self):
        pass

    def go(self, input):
        pass
