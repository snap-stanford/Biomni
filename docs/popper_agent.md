# POPPER Agent

## Overview

The `POPPERAgent` is a sophisticated hypothesis testing framework that combines POPPER's sequential falsification testing with biomni's rich biological tools and data resources for automatic hypothesis validation.

## Key Features

1. **Full Tool Ecosystem**: Access to all biomni tools including database queries, data analysis, and bioinformatics tools
2. **Unified Data Access**: Uses biomni's data lake and supports local data sources
3. **Sequential Falsification**: Implements POPPER's methodology for systematic hypothesis testing
4. **Advanced Code Generation**: Custom code generation for statistical tests

## Usage

```python
from biomni.agent.popper_agent import create_popper_agent

# Create the agent
agent = create_popper_agent(
    llm="claude-3-5-sonnet-20241022",
    data_path=None,  # Uses biomni data lake
    use_biomni=True,  # Enable full biomni capabilities
    max_num_of_tests=10,
    domain="biology"
)

# Test a hypothesis
hypothesis = "Gene X is associated with disease Y"
log, summary, result = agent.go(hypothesis)

print(f"Conclusion: {result['conclusion']}")
print(f"Summary: {summary}")
```

## Configuration Options

- `llm`: Language model to use (default: "claude-3-5-sonnet-20241022")
- `data_path`: Path to local data files (optional)
- `use_biomni`: Whether to use biomni's full tool ecosystem (default: True)
- `max_num_of_tests`: Maximum number of falsification tests (default: 10)
- `domain`: Scientific domain for specialized prompts (default: "biology")
- `alpha`: Significance level for hypothesis testing (default: 0.1)
- `aggregate_test`: Method for combining p-values ('Fisher', 'E-value', 'E-value_integral')

## How It Works

1. **Test Proposal**: The agent proposes falsification tests using domain expertise and available resources
2. **Test Implementation**: Custom code generation creates statistical tests using biomni tools
3. **Execution & Validation**: Code is executed with timeout and error handling
4. **Sequential Testing**: Multiple tests are aggregated using statistical methods
5. **Conclusion**: The hypothesis is accepted or rejected based on aggregated results

## Available Resources

### Data Lake
The agent has access to biomni's comprehensive data lake containing various biological datasets including:
- Gene expression data (GTEx, TCGA)
- Protein interaction networks
- Pathway databases
- Clinical datasets

### Tool Ecosystem
- **Database Tools**: Query UniProt, NCBI, KEGG, STRING, and other biological databases
- **Analysis Tools**: Statistical analysis, machine learning, bioinformatics pipelines
- **Visualization Tools**: Generate plots and charts for data exploration
- **Execution Environment**: Secure code execution with proper error handling

## Example Hypotheses

- "BRCA1 mutations are associated with increased breast cancer risk"
- "Gene expression of TP53 correlates with tumor grade in multiple cancer types"
- "Protein A interacts with protein B in DNA repair pathways"
- "Drug X targets proteins involved in cell cycle regulation"

## Implementation Details

### POPPERAgent Class
- **UnifiedDataLoader**: Handles both local and biomni data sources
- **EnhancedTestProposalAgent**: Generates scientifically sound test proposals
- **EnhancedTestCodingAgent**: Creates and executes statistical test code
- **Sequential Testing Framework**: Implements Fisher's method and E-value aggregation

### Workflow Components
- **LangGraph Integration**: Uses LangGraph for complex multi-step workflows
- **Tool Retrieval**: Dynamic selection of relevant tools based on hypothesis
- **Code Generation**: Structured code output with imports, logic, and validation
- **Error Recovery**: Automatic retry mechanisms and error handling

## Statistical Methods

### Aggregation Methods
- **Fisher's Method**: Combines p-values using chi-square distribution
- **E-value (Kappa)**: Uses kappa calibrator for e-value calculation
- **E-value (Integral)**: Uses integral calibrator for robust aggregation

### Validation Features
- **Relevance Checking**: Ensures test proposals are relevant to main hypothesis
- **Data Validation**: Prevents use of fake or fabricated data
- **P-value Extraction**: Robust extraction and validation of statistical results
- **Timeout Handling**: Prevents infinite execution loops

## Benefits

1. **Comprehensive**: Access to full biomni ecosystem
2. **Robust**: Advanced error handling and validation
3. **Flexible**: Supports multiple statistical methods and data sources
4. **Scientific**: Implements rigorous falsification testing methodology
5. **Scalable**: Handles complex multi-test scenarios efficiently
