import re

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


class FunctionGenerator(base_agent):
    """Agent that generates executable Python code scripts given a task description."""

    def __init__(self, llm="claude-3-7-sonnet-20250219", cheap_llm=None, temperature=0.7):
        """Initialize the PaperTaskExtractor agent.

        Args:
            llm (str): The LLM model to use
            cheap_llm (str, optional): A cheaper LLM for simpler tasks
        """
        super().__init__(llm, cheap_llm, temperature)
        self.log = []
        self.configure()

    def configure(self):
        """Configure the agent with appropriate prompts."""
        # Prompt for Python code generation
        self.system_prompt = """You are a senior Python engineer. Generate robust, idiomatic Python code that solves the user's task. Requirements:
        1. Output ONLY Python code, ideally inside a single triple-backtick code block.
        2. Include minimal inline comments and a small docstring.
        3. Add a `main()` and an `if __name__ == '__main__':` guard when appropriate.
        4. Avoid external dependencies unless necessary; if used, show `pip` installs in comments.
        5. Do not include prose before or after the code.
        6. When applicable, prioritize the use of codes on public repositories, such as HuggingFace or Github

        Generate Python codes for the following task:
        {task}
"""

    def _generate_code(self, task_description: str) -> str:
        """Generate codes given a task description.
        Args:
            task_description (str): task descriptions (possibly generated from previous steps)

        Returns:
            str: generated code string

        """
        prompt = self.system_prompt.format(task=task_description)
        message = self.llm.invoke(prompt)
        return message.content

    def _generate_script_filename(self, task_description: str, max_words: int = 6) -> str:
        """
        Generate a safe, meaningful Python script filename from a task description.
        Poised for update: may ask the agent to suggest meaningful names.

        Parameters:
        -----------
        task_description (str): task descriptions (possibly generated from previous steps)

        max_words : int
            Maximum number of words to include in the filename.

        Returns:
        --------
            str
            A lowercase, hyphen-free, safe filename ending in '.py'.
        """
        # Lowercase and remove non-alphanumeric (allow spaces for splitting)
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", task_description.lower())

        # Tokenize and select top words
        words = cleaned.split()
        selected_words = words[:max_words] if words else ["script"]

        # Join with underscores
        base_name = "_".join(selected_words)
        return f"{base_name}.py"

    def go(self, task_description: str):
        """Implement the inherited function to get the tasks done.

        Args:
            task_description (str): task descriptions (possibly generated from previous steps)

        Returns:
            tuple: (script_filename, results) where script_filename is a generated name for script file and results is the generated codes

        """
        self.log = []
        self.log.append(
            (
                "user",
                "Generate Python codes given a task description",
            )
        )

        script_filename = self._generate_script_filename(task_description)
        results = self._generate_code(task_description)
        return script_filename, self._extract_code_block(results)

    def _extract_code_block(self, s: str) -> str:
        """
        Extract the first fenced code block (``` or ```python) from s.
        """
        m = re.search(r"```(?:python)?\s*(.+?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None
