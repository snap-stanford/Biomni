import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base_CAi.agent.base_agent import base_agent


class PaperTaskExtractor(base_agent):
    """Agent that extracts generalizable tasks/experiments from academic papers.
    It processes papers in chunks and identifies common research tasks that could be shared across papers.
    """

    def __init__(
        self,
        llm="claude-3-7-sonnet-20250219",
        cheap_llm=None,
        tools=None,
        chunk_size=4000,
        chunk_overlap=400,
    ):
        """Initialize the PaperTaskExtractor agent.

        Args:
            llm (str): The LLM model to use
            cheap_llm (str, optional): A cheaper LLM for simpler tasks
            tools (list, optional): Any tools to use (not needed for this agent)
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks

        """
        super().__init__(llm, cheap_llm, tools)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.log = []
        self.configure()

    def configure(self):
        """Configure the agent with appropriate prompts."""
        # Prompt for analyzing paper chunks
        self.chunk_analysis_prompt = """You are a research methodology expert specializing in identifying computational tasks and data analysis procedures in academic papers.

Your job is to analyze chunks of academic papers and identify ONLY the most common, generalizable computational tasks that are widely used across biomedical research and can be implemented with Python or Linux code.

STRICT GUIDELINES:
1. ONLY extract tasks that are extremely common and standard in computational biomedical research
2. Each task MUST have clear, well-defined inputs and outputs
3. Tasks MUST be generalizable across many different papers and research questions
4. Be VERY selective - only include tasks that appear in hundreds of papers
5. If a task is specific to this paper, unclear, or not widely used, DO NOT include it
6. Focus on computational tasks that can be implemented with Python or Linux code
7. Each task should be something that could be implemented as a function with clear inputs/outputs
8. Also identify commonly used databases and software packages mentioned in the text
9. Tasks MUST be CONCRETE and SPECIFIC - include exact methodological details
10. Avoid vague task names like "Statistical Analysis" - instead use specific protocol names like "Two-way ANOVA with Tukey's Post-hoc Test using SciPy"
11. DO NOT include wet lab procedures that cannot be implemented computationally
12. ONLY include tasks that could be automated with code

For the following chunk of text from an academic paper, provide:
1. A list of ONLY the most common, generalizable COMPUTATIONAL tasks identified (be extremely selective)
2. For each task, clearly define:
   - Task name: A SPECIFIC and CONCRETE name with methodological details (e.g., "RNA-seq Differential Expression Analysis with DESeq2" instead of just "Gene Expression Analysis")
   - Input: What SPECIFIC data or parameters the task requires
   - Output: What SPECIFIC data or results the task produces
   - Code implementation: How this task could be implemented with Python or Linux code, including key libraries/packages
   - Frequency: How common this computational task is in biomedical research
   - Standard methods: The established computational techniques used to perform this task
   - Example: A brief description of how THIS specific paper uses this task (with specific details from the paper)
3. A list of commonly used databases mentioned in the text (if any)
4. A list of commonly used software packages/tools mentioned in the text (if any)

PAPER CHUNK:
{chunk_text}

Remember, it's better to return NO tasks than to include tasks that aren't extremely common, generalizable, and implementable with code. Quality over quantity is essential. Tasks MUST be CONCRETE with SPECIFIC methodological details and MUST be implementable with Python or Linux code.
"""

        # Prompt for consolidating tasks
        self.consolidation_prompt = """You are a research methodology expert. Your task is to consolidate lists of computational research tasks extracted from different chunks of an academic paper.

BE EXTREMELY SELECTIVE. Only include tasks that are:
1. Fundamental to computational biomedical research
2. Used in hundreds of papers across different subfields
3. Have clear, well-defined inputs and outputs
4. Represent standard computational approaches
5. Could be implemented as a function with specific inputs and outputs
6. Are CONCRETE and SPECIFIC with exact methodological details
7. Can be implemented with Python or Linux code
8. Are computational in nature, not wet lab procedures

REMOVE any tasks that:
- Are specific to a particular paper or dataset
- Lack clear inputs or outputs
- Are not widely used across biomedical research
- Are vague or poorly defined
- Represent niche or specialized techniques
- Have generic names without specific methodological details
- Cannot be implemented with code
- Require physical lab equipment or manual intervention

The output should be a JSON object with three main keys:
1. "tasks": A list of task objects, where each task has:
   - "task_name": A SPECIFIC and CONCRETE name with methodological details (e.g., "Single-cell RNA-seq Clustering with Seurat" instead of just "Cell Clustering")
   - "description": A clear description of what the computational task does
   - "inputs": SPECIFIC data types or parameters the task requires
   - "outputs": SPECIFIC data types or results the task produces
   - "code_implementation": How this task could be implemented with Python or Linux code, including key libraries/packages and a brief pseudocode example
   - "frequency": How common this computational task is in biomedical research
   - "standard_methods": The established computational techniques used to perform this task
   - "example": A specific example from THIS paper showing how the task was used (with concrete details)

2. "databases": A list of database objects, where each database has:
   - "name": The name of the database
   - "description": A brief description of what the database contains
   - "url": The URL of the database (if mentioned)
   - "usage": How the database is commonly used in computational biomedical research
   - "example": How this specific paper uses the database

3. "software": A list of software package objects, where each package has:
   - "name": The name of the software package
   - "description": A brief description of what the software does
   - "url": The URL or reference to the software (if mentioned)
   - "usage": How the software is commonly used in computational biomedical research
   - "example": How this specific paper uses the software

EXTRACTED INFORMATION FROM PAPER CHUNKS:
{task_lists}

Be ruthless in filtering - it's better to return a few truly common computational tasks than many that aren't universal or implementable with code.
Respond with only a valid JSON object containing the three lists described above.
"""

    def process_paper(self, paper_text: str) -> dict[str, list[dict[str, Any]]]:
        """Process a paper and extract generalizable tasks/experiments, databases, and software.

        Args:
            paper_text (str): The full text of the paper

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with tasks, databases, and software

        """
        # Split the paper into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_text(paper_text)

        # Process each chunk to extract tasks
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
            chunk_tasks = self._process_chunk(chunk)
            chunk_results.append(chunk_tasks)

        # Consolidate tasks from all chunks
        consolidated_results = self._consolidate_tasks(chunk_results)
        return consolidated_results

    def _process_chunk(self, chunk_text: str) -> str:
        """Process a single chunk of the paper to extract tasks.

        Args:
            chunk_text (str): The chunk text to process

        Returns:
            str: Extracted tasks from this chunk

        """
        prompt = self.chunk_analysis_prompt.format(chunk_text=chunk_text)
        message = self.llm.invoke(prompt)
        return message.content

    def _consolidate_tasks(self, chunk_results: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Consolidate tasks, databases, and software extracted from different chunks into a unified structure.

        Args:
            chunk_results (List[str]): Results from each chunk

        Returns:
            Dict[str, List[Dict[str, Any]]]: Consolidated information with tasks, databases, and software

        """
        # Combine all chunk results
        all_tasks = "\n\n===== CHUNK SEPARATOR =====\n\n".join(chunk_results)

        # Use the consolidation prompt to merge and organize tasks
        prompt = self.consolidation_prompt.format(task_lists=all_tasks)
        response = self.llm.invoke(prompt)

        # Extract the JSON from the response
        try:
            # Try to parse the entire response as JSON
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            try:
                # Look for JSON-like content between triple backticks
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response.content)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Fallback: just return the text response
                    return {
                        "tasks": [
                            {
                                "error": "Could not parse JSON",
                                "raw_response": response.content,
                            }
                        ],
                        "databases": [],
                        "software": [],
                    }
            except Exception:
                return {
                    "tasks": [
                        {
                            "error": "Could not parse JSON",
                            "raw_response": response.content,
                        }
                    ],
                    "databases": [],
                    "software": [],
                }

        # Ensure the result has the expected structure
        if not isinstance(result, dict):
            result = {
                "tasks": result if isinstance(result, list) else [],
                "databases": [],
                "software": [],
            }

        # Ensure all required keys exist
        for key in ["tasks", "databases", "software"]:
            if key not in result:
                result[key] = []

        return result

    def go(self, paper_text: str):
        """Process a paper and return the extracted tasks, databases, and software.

        Args:
            paper_text (str): The full text of the paper

        Returns:
            tuple: (log, results) where log is a list of processing steps and results is the final result

        """
        self.log = []
        self.log.append(
            (
                "user",
                "Extract only the most common and generalizable biomedical research tasks, databases, and software from this paper",
            )
        )

        results = self.process_paper(paper_text)

        result_str = json.dumps(results, indent=2)
        self.log.append(("assistant", result_str))

        return self.log, results

    def save_results(self, results: dict[str, list[dict[str, Any]]], output_path: str):
        """Save the extracted tasks, databases, and software to a JSON file.

        Args:
            results (Dict[str, List[Dict[str, Any]]]): The extracted tasks, databases, and software
            output_path (str): Path to save the results

        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    def result_formatting(self, output_class, task_intention):
        """Format the results according to a specific output class.

        Args:
            output_class: The class to format the output as
            task_intention: Description of the task

        Returns:
            The formatted result

        """
        format_check_prompt = ChatPromptTemplate.from_messages(
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

        checker_llm = format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result
