# Human-in-the-Loop Mode for Biomni Agent

The Biomni agent now supports **Human-in-the-Loop (HITL)** functionality, allowing users to interactively confirm, edit, or reject the agent's plans before execution. This feature provides enhanced control and oversight over the agent's decision-making process.

## 🎯 Key Features

- **Interactive Plan Confirmation**: Review and approve code execution and final solutions
- **Real-time Plan Editing**: Modify the agent's proposed code or solutions before execution
- **Flexible Control**: Approve, edit, reject, or stop execution at any point
- **Safety & Oversight**: Prevent unwanted actions through human confirmation
- **Educational Value**: Understand the agent's reasoning and learn from its approach

## 🚀 Usage

### Basic Interactive Mode

```python
from biomni.agent.a1 import A1

# Create an agent with interactive mode enabled
agent = A1(
    path="./data",
    llm="gpt-4o-mini",
    source="OpenAI",
    interactive=True  # Enable human-in-the-loop mode
)

# Execute a query - you'll be prompted for confirmation
log, response = agent.go("Create a scatter plot of random data")
```

### Non-Interactive Mode (Default)

```python
# Standard agent behavior - automatic execution
agent = A1(
    path="./data",
    llm="gpt-4o-mini",
    source="OpenAI",
    interactive=False  # Default: no human confirmation
)

log, response = agent.go("Create a scatter plot of random data")  # Executes automatically
```

## 🤝 Interactive Workflow

When interactive mode is enabled, the agent will pause before:

1. **Code Execution**: Before running any `<execute>` blocks
2. **Final Solutions**: Before providing `<solution>` responses

At each confirmation point, you have four options:

### 1. ✅ Approve and Proceed
- Accepts the proposed plan as-is
- Continues with execution

### 2. ✏️ Edit the Plan
- Allows you to modify the code or solution
- Opens an editor interface for making changes
- Proceeds with your modified version

### 3. ❌ Reject and Regenerate
- Rejects the current plan
- Asks the agent to generate an alternative approach
- Useful when the approach isn't suitable

### 4. 🛑 Stop Execution
- Immediately stops the agent
- Useful for safety or when you want to start over

## 📋 Example Interactive Session

```
🤖 BIOMNI AGENT - CODE EXECUTION CONFIRMATION
============================================================

📋 Generated code execution:
----------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.randn(100)
y = np.random.randn(100)

# Create scatter plot
plt.scatter(x, y)
plt.title('Random Data Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()
----------------------------------------

🤔 What would you like to do?
  1. ✅ Approve and proceed
  2. ✏️  Edit the plan
  3. ❌ Reject and ask agent to regenerate
  4. 🛑 Stop execution

Enter your choice (1-4): 2

✏️  Edit mode - Current code execution:
----------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.randn(100)
y = np.random.randn(100)

# Create scatter plot
plt.scatter(x, y)
plt.title('Random Data Scatter Plot')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()
----------------------------------------

Enter your modifications (press Enter twice to finish):
# Add color and transparency
plt.scatter(x, y, alpha=0.6, c='blue')
plt.grid(True)


✅ Plan updated! Proceeding with modified version...
```

## 🎓 Use Cases

### Research & Analysis
- **Exploratory Data Analysis**: Review analysis steps before execution
- **Scientific Computing**: Ensure computational approaches are appropriate
- **Data Visualization**: Fine-tune plots and charts before generation

### Educational Purposes
- **Learning AI Reasoning**: See how the agent approaches problems
- **Code Review**: Understand and improve generated code
- **Best Practices**: Learn from AI-generated solutions

### Production & Safety
- **Mission-Critical Tasks**: Human oversight for important decisions
- **Data Security**: Review data access and manipulation
- **Compliance**: Ensure adherence to organizational policies

### Collaborative Development
- **Domain Expertise**: Inject specialized knowledge into AI workflows
- **Quality Assurance**: Review outputs before finalization
- **Iterative Refinement**: Gradually improve solutions through feedback

## ⚙️ Configuration Options

The interactive mode is controlled by the `interactive` parameter in the A1 constructor:

```python
agent = A1(
    path="./data",
    llm="gpt-4o-mini",
    source="OpenAI",
    interactive=True,  # Enable/disable interactive mode
    use_tool_retriever=True,
    timeout_seconds=600,
    commercial_mode=False
)
```

When interactive mode is enabled, you'll see additional configuration output:

```
🤝 INTERACTIVE MODE: Enabled
  • Human-in-the-loop confirmation for code execution
  • Plan editing capabilities before execution
  • User control over agent decisions
```

## 🔧 Integration with Existing Workflows

### Streaming Mode
Interactive mode works with both `go()` and `go_stream()` methods:

```python
# Interactive streaming
for step in agent.go_stream("Analyze this dataset"):
    print(step["output"])
    # Confirmation prompts will appear during streaming
```

### Error Handling
Interactive mode gracefully handles interruptions:

```python
try:
    log, response = agent.go("Complex analysis task")
except KeyboardInterrupt:
    print("User stopped execution")
```

### Non-Interactive Fallback
Easily switch between modes for different use cases:

```python
# Interactive for development
dev_agent = A1(interactive=True, ...)
log, response = dev_agent.go("Prototype solution")

# Non-interactive for production
prod_agent = A1(interactive=False, ...)
log, response = prod_agent.go("Production analysis")
```

## 🔮 Future Integration

This human-in-the-loop implementation is designed for easy integration with:

- **OpenWebUI**: Web-based confirmation interfaces
- **Langflow**: Visual workflow builders with approval nodes
- **Custom UIs**: RESTful APIs for confirmation endpoints
- **Chat Interfaces**: Interactive messaging platforms

The modular design allows the confirmation logic to be easily replaced with web-based or API-driven alternatives while maintaining the same core functionality.

## 🧪 Testing

Run the test suite to verify human-in-the-loop functionality:

```bash
# Automated tests
python tests/test_human_in_the_loop.py

# Interactive example
python examples/human_in_the_loop_example.py
```

## 💡 Tips

1. **Start Simple**: Begin with basic queries to understand the confirmation flow
2. **Edit Incrementally**: Make small changes during editing to avoid errors
3. **Use Rejection Wisely**: Reject plans when the approach is fundamentally wrong
4. **Learn from Patterns**: Observe how the agent approaches different problem types
5. **Combine Modes**: Use interactive for development, non-interactive for production
