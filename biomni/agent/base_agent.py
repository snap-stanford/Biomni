class base_agent:
    def __init__(self, llm=None, cheap_llm=None, tools=None):
        self.llm = llm
        self.cheap_llm = cheap_llm
        self.tools = tools