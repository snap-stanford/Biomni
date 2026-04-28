from langchain_core.prompts import ChatPromptTemplate

from base_CAi.llm import get_llm


class qa_llm:
    def __init__(self, path="./data", llm="claude-3-haiku-20240307", lab_bench_reproduce=False):
        self.path = path
        self.llm = get_llm(llm)

        if lab_bench_reproduce:
            self.prompt_modifier = """
The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Think step by step. \n
            """
        else:
            self.prompt_modifier = ""
        self.log = []

    def configure(self):
        pass

    def go(self, input):
        self.log = []
        self.log.append(("user", input))
        message = self.llm.invoke(self.prompt_modifier + input)
        self.log.append(("assistant", message.content))
        return [message.content], message.content

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result
