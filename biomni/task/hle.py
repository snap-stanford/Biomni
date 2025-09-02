import numpy as np
import pandas as pd

from biomni.task.base_task import base_task

np.random.seed(42)


def shuffle(x):
    np.random.shuffle(x)
    return x


class humanity_last_exam(base_task):
    def __init__(self, path="./data", category="Biology/Medicine", answer_type="multipleChoice"):
        if category not in [
            "Other",
            "Humanities/Social Science",
            "Math",
            "Physics",
            "Computer Science/AI",
            "Biology/Medicine",
            "Chemistry",
            "Engineering",
        ]:
            raise ValueError(
                "category must be one of ['Other', 'Humanities/Social Science', 'Math', 'Physics', 'Computer Science/AI', 'Biology/Medicine', 'Chemistry', 'Engineering']"
            )
        if answer_type not in ["exactMatch", "multipleChoice"]:
            raise ValueError("answer_type must be one of ['exactMatch' or 'multipleChoice']")

        self.dataset = category  # Store dataset type
        self.answer_type = answer_type
        df = pd.read_parquet(path + "/hle/test_sampled_biology_medicine.parquet")

        # Extract answer choices from question text
        def extract_options(question):
            # Find the "Answer Choices:" section
            if "Answer Choices:" not in question:
                return []
            choices = question.split("Answer Choices:")[1].strip()

            # Split on A., B., C. etc and clean up
            options = []
            letters = [
                "A.",
                "B.",
                "C.",
                "D.",
                "E.",
                "F.",
                "G.",
                "H.",
                "I.",
                "J.",
                "K.",
                "L.",
                "M.",
                "N.",
                "O.",
                "P.",
                "Q.",
                "R.",
                "S.",
                "T.",
                "U.",
                "V.",
                "W.",
                "X.",
                "Y.",
                "Z.",
            ]
            for i, letter in enumerate(letters):
                if letter in choices:
                    # Define the next letter if available
                    next_letter = letters[i + 1] if i + 1 < len(letters) else None

                    # Split between current letter and next letter
                    parts = choices.split(letter)[1]
                    if next_letter and next_letter in parts:
                        option = parts.split(next_letter)[0].strip()
                    else:
                        option = parts.strip()

                    options.append(option)
            return options

        def extract_question(question):
            return question.split("Answer Choices:")[0].strip()

        # Extract options and answers only for multiple choice questions and category is the same as the dataset
        df = df[df["category"] == self.dataset]
        df = df[df["answer_type"] == "multipleChoice"]
        df["question_text"] = df.question
        df["letter_answer"] = df["answer"].apply(lambda x: x[0])

        self.query = df.question_text.values
        # self.options = df.options_letters.values
        self.answer = df.letter_answer.values

        self.prompt = """Question: {question}"""

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))

        return {
            "prompt": self.prompt.format(
                question=self.query[index],
                # options = self.options[index]
            ),
            "answer": self.answer[index],
        }

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score

        ground_truth = self.answer
        response = np.array(response)

        return {
            "accuracy": accuracy_score(ground_truth, response),
            "coverage": np.mean(response != self.refrain_label),
            "refrain_ratio": np.mean(response == self.refrain_label),
            "precision": accuracy_score(
                ground_truth[np.where(response != self.refrain_label)],
                response[np.where(response != self.refrain_label)],
            ),
        }

    def output_class(self):
        from pydantic import BaseModel, Field

        class MultipleChoiceOutput(BaseModel):
            """Multiple choice output."""

            choice: str | None = Field(
                description="Multiple choice answer. For example, if there is <answer>A</answer> in the prompt, the output should be 'A'."
            )

        return MultipleChoiceOutput
