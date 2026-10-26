# DETAILS.md

🔍 **Powered by [Detailer](https://detailer.ginylil.com)** - Context-aware codebase analysis

---

## 1. Project Overview

### Project Purpose & Domain

This project is a comprehensive **biomedical AI toolkit and research platform** designed to facilitate **biomedical data analysis, knowledge extraction, and AI-driven reasoning**. It integrates large language models (LLMs), domain-specific bioinformatics tools, and scientific data processing pipelines to enable:

- Automated extraction of biomedical knowledge from literature (e.g., bioRxiv papers)
- Querying and integration of diverse biomedical databases and APIs
- Execution of domain-specific computational biology and physiology analyses
- AI agent orchestration for complex biomedical reasoning and tool invocation
- Benchmarking and evaluation of biomedical tasks and datasets

### Target Users and Use Cases

- **Biomedical researchers and data scientists** seeking to automate literature mining, data retrieval, and analysis workflows.
- **Bioinformaticians** requiring integrated access to multiple biological databases and computational tools.
- **AI researchers** interested in applying LLMs and autonomous agents to biomedical problem solving.
- **Developers and integrators** building domain-specific AI pipelines and scientific workflows.
- Use cases include:
  - Extracting structured biomedical tasks and entities from scientific papers
  - Querying gene, protein, disease, and pathway databases via natural language prompts
  - Running computational models of biological systems (e.g., metabolic networks, signaling)
  - Performing image analysis and quantitative pathology workflows
  - Orchestrating multi-step AI reasoning with tool use and self-criticism

### Core Business Logic and Domain Models

- **Biomedical domain models**: gene IDs, protein structures, pathways, disease ontologies, experimental assays.
- **Task abstractions**: benchmark tasks with prompt/response evaluation (e.g., humanity last exam, lab bench).
- **Tool metadata schemas**: declarative descriptions of biomedical tools and APIs for dynamic invocation.
- **AI agent workflows**: ReAct-style reasoning graphs integrating LLMs, tool calls, retrieval, and self-critique.
- **Data models**: structured JSON, pandas DataFrames, numpy arrays representing biological data and analysis results.

---

## 2. Architecture and Structure

### High-Level Architecture

The system is organized into modular layers and components:

- **Core Library (`biomni/`)**: Contains main application logic, including:
  - **Agent framework (`biomni/agent/`)**: Implements autonomous AI agents using LLMs and workflow graphs.
  - **Task definitions (`biomni/task/`)**: Abstract base and concrete biomedical benchmark tasks.
  - **Tool implementations (`biomni/tool/`)**: Domain-specific analysis functions, API clients, and computational biology workflows.
  - **Tool metadata (`biomni/tool/tool_description/`)**: Declarative schemas describing tool APIs and parameters.
  - **Model components (`biomni/model/`)**: AI-driven resource retriever for selecting relevant tools and data.
  - **Utility modules (`biomni/utils.py`, `biomni/llm.py`, `biomni/env_desc.py`)**: Helpers for LLM instantiation, system commands, environment descriptions.
  - **Versioning (`biomni/version.py`)**: Package version management.

- **Environment Setup (`biomni_env/`)**: Scripts and configuration files for reproducible environment provisioning, including:
  - Conda environment YAMLs (`environment.yml`, `bio_env.yml`)
  - R package specifications (`r_packages.yml`)
  - CLI tools installer (`install_cli_tools.sh`)
  - Shell scripts for environment setup (`setup.sh`, `setup_path.sh`)

- **Scripts (`biomni/biorxiv_scripts/`)**: Data processing pipelines for literature mining and task extraction.

- **Documentation and Configuration**:
  - Root-level files: `README.md`, `CONTRIBUTION.md`, `pyproject.toml`, `.pre-commit-config.yaml`.

---

### Complete Repository Structure

```
.
├── biomni/ (90 items)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── a1.py
│   │   ├── env_collection.py
│   │   ├── qa_llm.py
│   │   └── react.py
│   ├── biorxiv_scripts/
│   │   ├── extract_biorxiv_tasks.py
│   │   ├── generate_function.py
│   │   └── process_all_subjects.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── task/
│   │   ├── __init__.py
│   │   ├── base_task.py
│   │   ├── hle.py
│   │   └── lab_bench.py
│   ├── tool/ (65 items)
│   │   ├── schema_db/ (25 items)
│   │   │   ├── cbioportal.pkl
│   │   │   ├── clinvar.pkl
│   │   │   ├── dbsnp.pkl
│   │   │   ├── emdb.pkl
│   │   │   ├── ensembl.pkl
│   │   │   ├── geo.pkl
│   │   │   ├── gnomad.pkl
│   │   │   ├── gtopdb.pkl
│   │   │   ├── gwas_catalog.pkl
│   │   │   ├── interpro.pkl
│   │   │   └── ... (15 more files)
│   │   ├── tool_description/ (18 items)
│   │   │   ├── biochemistry.py
│   │   │   ├── bioengineering.py
│   │   │   ├── biophysics.py
│   │   │   ├── cancer_biology.py
│   │   │   ├── cell_biology.py
│   │   │   ├── database.py
│   │   │   ├── genetics.py
│   │   │   ├── genomics.py
│   │   │   ├── immunology.py
│   │   │   ├── literature.py
│   │   │   ├── microbiology.py
│   │   │   ├── molecular_biology.py
│   │   │   ├── pathology.py
│   │   │   ├── pharmacology.py
│   │   │   ├── physiology.py
│   │   │   ├── support_tools.py
│   │   │   ├── synthetic_biology.py
│   │   │   └── systems_biology.py
│   │   ├── __init__.py
│   │   ├── biochemistry.py
│   │   ├── bioengineering.py
│   │   ├── biophysics.py
│   │   ├── cancer_biology.py
│   │   ├── cell_biology.py
│   │   ├── database.py
│   │   ├── genetics.py
│   │   └── ... (12 more files)
│   ├── __init__.py
│   ├── env_desc.py
│   ├── llm.py
│   ├── utils.py
│   └── version.py
├── biomni_env/ (9 items)
│   ├── README.md
│   ├── bio_env.yml
│   ├── cli_tools_config.json
│   ├── environment.yml
│   ├── install_cli_tools.sh
│   ├── install_r_packages.R
│   ├── r_packages.yml
│   ├── setup.sh
│   └── setup_path.sh
├── figs/
│   └── biomni_logo.png
├── tutorials/
│   ├── examples/
│   │   └── cloning.ipynb
│   ├── 101_biomni.ipynb
│   └── biomni_101.ipynb
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTION.md
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## 3. Technical Implementation Details

### Core Modules and Their Roles

#### `biomni/agent/`

- Implements autonomous AI agents using the **ReAct paradigm**:
  - `react.py`: Main ReAct agent class managing reasoning, tool invocation, retrieval, and self-criticism workflows.
  - `env_collection.py`: Environment and data retrieval utilities.
  - `qa_llm.py`: Question-answering LLM wrappers.
  - `a1.py`: Possibly experimental or auxiliary agent code.

- Uses **langgraph** for workflow graph orchestration and **langchain** for LLM integration.

#### `biomni/task/`

- Defines **benchmark tasks** with a common interface:
  - `base_task.py`: Abstract base class specifying methods like `get_example()`, `evaluate()`, `output_class()`.
  - `hle.py`: "Humanity Last Exam" task implementation.
  - `lab_bench.py`: Lab bench dataset task.

- Tasks load data (e.g., parquet files), generate prompts, and evaluate LLM responses.

#### `biomni/tool/`

- Contains **domain-specific scientific analysis functions** organized by subdomains:
  - `biochemistry.py`, `bioengineering.py`, `biophysics.py`, `cancer_biology.py`, `cell_biology.py`, `genetics.py`, `pathology.py`, `physiology.py`, `systems_biology.py`, etc.
  - Each file implements multiple functions performing analyses, simulations, or data processing workflows.
  - Functions accept input files/parameters and return detailed textual logs and output files.

- **API client modules** (e.g., `database.py`) provide facade functions to query external biomedical databases (UniProt, GWAS Catalog, Ensembl, etc.) via REST or GraphQL APIs, often using LLMs to generate query payloads from natural language prompts.

- **Tool registry (`tool_registry.py`)** manages metadata about available tools, supporting dynamic registration and lookup.

#### `biomni/tool/tool_description/`

- Contains **declarative metadata schemas** describing tool APIs:
  - Each file exports a `description` list of dictionaries defining tool names, descriptions, required and optional parameters with types and defaults.
  - Supports **dynamic API generation, validation, and documentation**.
  - Organized by biological domain (e.g., genetics, immunology, pathology).

#### `biomni/model/retriever.py`

- Implements `ToolRetriever` class for **AI-driven resource selection**:
  - Uses LLMs (OpenAI or Anthropic) to parse user queries and select relevant tools, datasets, and libraries.
  - Encapsulates prompt formatting and response parsing logic.

#### `biomni/utils.py` and `biomni/llm.py`

- `utils.py`: Utility functions for running system commands (R, Bash), file operations, schema generation, logging, and colorized printing.
- `llm.py`: Factory functions to instantiate LLMs (OpenAI, Anthropic) with configurable parameters.

#### `biomni/env_desc.py`

- Contains **environment and dataset descriptions**, acting as a centralized metadata repository for datasets and experimental environments.

---

### Environment Setup (`biomni_env/`)

- `setup.sh`: Main shell script to create conda environment, install R packages, and CLI bioinformatics tools.
- `install_cli_tools.sh`: Automates downloading, compiling, and installing external bioinformatics command-line tools, managing PATH and verification.
- `r_packages.yml`: Lists R packages required.
- `environment.yml` and `bio_env.yml`: Conda environment specifications.
- `setup_path.sh`: Shell script to update environment variables for CLI tools.

---

### Entry Points and Execution Flow

- **Agent usage**: Instantiate `react` agent from `biomni.agent.react`, configure with tools and retrieval, then call `go(prompt)` to run reasoning workflows.
- **Task evaluation**: Use classes in `biomni.task` to load datasets, generate prompts, and evaluate LLM outputs.
- **Tool invocation**: Call functions in `biomni.tool` modules or use API facades in `database.py` to query external resources.
- **Metadata-driven tool discovery**: Use `tool_registry.py` and `tool_description` schemas to dynamically discover and validate tools.
- **Environment setup**: Run `biomni_env/setup.sh` to provision environment and install dependencies.

---

## 4. Development Patterns and Standards

### Code Organization Principles

- **Modular design**: Clear separation of concerns by domain and functionality (agent, task, tool, model).
- **Functional programming style**: Most analysis modules use standalone functions with explicit inputs and outputs.
- **Declarative metadata**: Tool descriptions and schemas are separated from implementation, enabling dynamic validation and UI generation.
- **Abstract base classes**: Used in `biomni.task.base_task` to enforce consistent task interfaces.
- **Factory pattern**: Used in `llm.py` to instantiate LLMs based on configuration.
- **Strategy pattern**: Task implementations and tool retrieval use interchangeable strategies.

### Testing and Coverage

- No explicit test files detected; testing likely manual or via notebooks (`tutorials/`).
- Tasks and tools return detailed logs suitable for manual verification.
- Metadata schemas facilitate automated validation of inputs.

### Error Handling and Logging

- Use of try-except blocks around external calls and subprocesses.
- Logging via custom callback handlers (`PromptLogger`, `NodeLogger`) in LLM interactions.
- Utilities provide colorized printing and error wrappers for robustness.

### Configuration Management

- Environment variables for API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
- YAML and JSON files for environment and tool configuration.
- Dynamic loading of schemas from pickle files for API request generation.
- CLI tools and R packages installed via scripted environment setup.

---

## 5. Integration and Dependencies

### External Libraries

- **LLM & AI Frameworks**: `langchain_core`, `langchain_openai`, `langchain_anthropic`
- **Scientific Computing**: `numpy`, `pandas`, `scipy`, `scikit-image`, `matplotlib`, `BioPython`, `cobra`, `sklearn`
- **Data Processing**: `pickle`, `json`, `requests`, `PyPDF2`
- **System and OS**: `subprocess`, `os`, `sys`, `tempfile`, `multiprocessing`
- **Others**: `tqdm` (progress bars), `enum`, `ast` (code introspection)

### External APIs and Data Sources

- Biomedical databases: UniProt, GWAS Catalog, Ensembl, ClinVar, dbSNP, EMDB, GEO, GnomAD, InterPro, etc.
- Bioinformatics tools: PLINK, IQ-TREE, GCTA, MACS2, samtools, LUMPY, installed via CLI tools installer.
- R packages for statistical and bioinformatics analyses.

### Build and Deployment Dependencies

- Python 3 environment managed via Conda (`environment.yml`).
- R environment with specified packages (`r_packages.yml`).
- Shell scripts for CLI tool installation and environment setup.
- Pre-commit hooks for code quality and security.

---

## 6. Usage and Operational Guidance

### Getting Started

1. **Environment Setup**
   - Run `biomni_env/setup.sh` to create the Conda environment, install R packages, and CLI tools.
   - Source `biomni_env/setup_path.sh` or add it to your shell profile to configure PATH.

2. **API Keys**
   - Set environment variables `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` for LLM access.

3. **Running Agents**
   - Import and instantiate the `react` agent from `biomni.agent.react`.
   - Configure with desired tools and retrieval options.
   - Call `go(prompt)` to execute reasoning workflows.

4. **Executing Tasks**
   - Use classes in `biomni.task` to load datasets and evaluate LLM responses.
   - Implement new tasks by subclassing `base_task` and following the interface.

5. **Querying Databases**
   - Use `biomni.tool.database` functions (e.g., `query_uniprot`, `query_gwas_catalog`) to retrieve data via natural language or direct parameters.

6. **Extending Tools**
   - Add new tool metadata in `biomni/tool/tool_description/` as structured dictionaries.
   - Implement corresponding analysis functions in `biomni/tool/`.
   - Register tools in `tool_registry.py` for discovery.

### Monitoring and Debugging

- Use logging callbacks (`PromptLogger`, `NodeLogger`) to trace LLM interactions.
- Check output logs returned by analysis functions for detailed execution info.
- Use pre-commit hooks to maintain code quality.

### Performance and Scalability

- Modular design allows parallel execution of tasks and tools.
- Timeout wrappers in agent tools prevent hanging executions.
- Use of efficient numerical libraries (`numpy`, `scipy`) for computational tasks.
- Large data handled via streaming and chunking (e.g., PDF text extraction).

### Security Considerations

- API keys managed via environment variables, not hardcoded.
- Pre-commit hooks include security checks.
- External tool installations verified via version commands.

### Observability

- Progress bars (`tqdm`) used in data processing scripts.
- Structured logs and JSON outputs facilitate downstream analysis.
- Agent workflows produce detailed message histories for audit.

---

## Summary

This project is a **modular, extensible biomedical AI platform** integrating **LLM-powered agents**, **domain-specific scientific tools**, and **metadata-driven APIs** to automate complex biomedical research workflows. It emphasizes **declarative tool descriptions**, **dynamic resource retrieval**, and **robust environment provisioning** to enable researchers and developers to build, evaluate, and extend AI-driven biomedical applications efficiently.

---

# End of DETAILS.md
