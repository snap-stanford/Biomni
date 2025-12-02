# Workflow Generation Module

This module provides functionality for generating workflow recommendations from various sources including data files and research papers.

## Installation

Make sure you have the required dependencies installed:
```bash
pip install pydantic langchain-core pillow pandas
# For PDF processing:
pip install PyMuPDF  # or pdfplumber, or pypdf
```

## Quick Start

### Basic Usage

```python
from biomni.sop import WorkflowGenerator

# Initialize the generator
generator = WorkflowGenerator()

# Generate workflow from data files
file_paths = ["data/your_data.xlsx"]
workflow_recommendations = generator.generate_workflow_recommendation(
    file_paths=file_paths,
    num_prompts=1
)
print(workflow_recommendations[0])
```

### Generate Multiple Workflow Variations

```python
# Generate 3 different workflow variations in parallel
workflow_recommendations = generator.generate_workflow_recommendation(
    file_paths=file_paths,
    num_prompts=3
)

for i, workflow in enumerate(workflow_recommendations, 1):
    print(f"Workflow Variation {i}:")
    print(workflow)
```

### Generate Workflow from Research Paper

```python
# Extract workflow from paper
workflow = generator.extract_workflow_from_paper(
    pdf_path="paper.pdf",
    mode="integrated"  # or "methods_only", "results_only"
)

# Generate workflow based on paper
workflow_recommendations = generator.generate_workflow_from_paper(
    pdf_path="paper.pdf",
    mode="integrated",
    num_prompts=1
)
```

### Generate Workflow from Paper + Data Files

```python
# Combine paper analysis with data file context
workflow_recommendations = generator.generate_workflow_from_paper(
    pdf_path="paper.pdf",
    file_paths=["data1.xlsx", "data2.csv"],
    mode="integrated",
    num_prompts=1
)
```

## Supported File Types

### Data Files
- **Excel files**: `.xlsx`, `.xls`
- **Text files**: `.txt`, `.csv`, `.tsv`, `.json`, `.xml`, `.html`, `.md`, `.log`, `.py`, `.r`, `.sh`, `.yaml`, `.yml`, `.ini`, `.cfg`, `.conf`
- **Image files**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.tif`, `.webp`

### Research Papers
- **PDF files**: Extracted using PyMuPDF, pdfplumber, or pypdf

## API Reference

### WorkflowGenerator

#### `__init__(prompts_dir=None)`
Initialize the WorkflowGenerator.

**Parameters:**
- `prompts_dir` (str, optional): Directory containing prompt template files

#### `generate_workflow_recommendation(file_paths, num_prompts=1)`
Generate workflow recommendations for given file paths.

**Parameters:**
- `file_paths` (str or List[str]): File path(s) to analyze
- `num_prompts` (int): Number of workflow variations to generate (default: 1)

**Returns:**
- `List[str]`: List of workflow recommendation strings

#### `extract_workflow_from_paper(pdf_path, mode="integrated")`
Extract analysis workflow from PDF paper.

**Parameters:**
- `pdf_path` (str): Path to PDF file
- `mode` (str): Extraction mode - "integrated", "methods_only", or "results_only"

**Returns:**
- `str`: Structured workflow description

#### `generate_workflow_from_paper(pdf_path, file_paths=None, mode="integrated", num_prompts=1)`
Generate workflow recommendations based on research paper and optional data files.

**Parameters:**
- `pdf_path` (str): Path to research paper PDF
- `file_paths` (str or List[str], optional): Data file paths to analyze
- `mode` (str): Paper extraction mode (default: "integrated")
- `num_prompts` (int): Number of workflow variations to generate (default: 1)

**Returns:**
- `List[str]`: List of workflow recommendation strings

## Examples

See the test files for more examples:
- `test_workflow_generator.py` - Comprehensive test suite

## Configuration

The module requires LangSmith configuration for tracing (optional):

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "your_project_name"
```

## Notes

- The module uses Claude Haiku 4.5 model by default for workflow generation
- Parallel generation is automatically handled using LangChain's batch method
- Image files are encoded as base64 and sent to the LLM for vision-based analysis
- Excel files show preview of first 5 rows and column information
- Text files show preview of first 5 lines or 1000 characters
