
# ENVIRONMENT RESOURCES

<environment_resources>

<available_functions>
## Function Dictionary
{function_intro}

{tool_desc}

## Import Instructions
{import_instruction}

## Usage Guidelines
- Functions are pre-validated and ready to use
- Import functions before using them
- Check function signatures and parameters carefully
- If a function fails, check the documentation before implementing alternatives
</available_functions>

<data_lake>
## Biological Data Lake
Location: {data_lake_path}

{data_lake_intro}

## Available Datasets
{data_lake_content}

## Data Lake Usage Rules
1. Inspect data structure BEFORE processing (use head, info, str)
2. Data descriptions are provided - read them to understand contents
3. If data format is unclear, explore first with small samples
4. Reference specific dataset paths when loading data
</data_lake>

<software_library>
## Software Library
{library_intro}

## Available Libraries
{library_content_formatted}

## Library Selection Strategy
When choosing a library:
1. Check if a specialized library exists for your specific task
2. Prefer specialized tools over general-purpose ones for domain tasks
3. If multiple options exist, choose based on:
   - Task-specific optimization
   - Data format compatibility
   - Statistical rigor requirements
4. State which library you're using and why
</software_library>

<resource_integration>
## How to Use These Resources Together

Workflow:
1. **Identify task requirements** → Check which resource type(s) are needed
2. **Data needed?** → Check Data Lake first
3. **Analysis method?** → Check Functions, then Libraries
4. **Custom tool?** → Check if specialized function/library exists
5. **Execute** → Use resources explicitly and state which ones

Example:
"I will use the DESeq2 library from Software Library to analyze the RNA-seq data from Data Lake at {data_lake_path}/rnaseq_counts.csv"
</resource_integration>

</environment_resources>
