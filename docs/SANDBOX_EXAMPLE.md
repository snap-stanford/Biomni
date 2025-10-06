# Biomni Sandbox Mode Example

This example demonstrates how to use the new sandbox mode feature in Biomni.

## Basic Usage

```python
from biomni.agent import A1

# Enable sandbox mode with auto-generated session folder
agent = A1(
    path='./data',
    sandbox_mode=True,  # Enable sandbox mode
    commercial_mode=True
)

# The agent will automatically create a sandbox directory like:
# sandbox/session_20251006_143022/

# All file operations in Python code will happen in the sandbox
result = agent.go("""
Create a simple analysis and save the results to a CSV file.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'score': [95, 87, 92, 88]
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('analysis_results.csv', index=False)
print("Data saved to analysis_results.csv")

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['score'])
plt.xlabel('Age')
plt.ylabel('Score')
plt.title('Age vs Score Analysis')
plt.savefig('analysis_plot.png')
print("Plot saved to analysis_plot.png")

# List files in current directory
import os
print(f"Files created: {os.listdir('.')}")
```
""")

# Check where files were created
sandbox_path = agent.get_sandbox_path()
print(f"All files were created in: {sandbox_path}")
```

## Custom Sandbox Path

```python
# Use a custom sandbox directory
agent = A1(
    path='./data',
    sandbox_mode=True,
    sandbox_path='/tmp/my_analysis_workspace',  # Custom path
    commercial_mode=True
)

# All file operations will happen in /tmp/my_analysis_workspace/
result = agent.go("Create some analysis files...")
```

## Regular Mode (No Sandbox)

```python
# Disable sandbox mode (default behavior)
agent = A1(
    path='./data',
    sandbox_mode=False,  # Or omit this parameter (default is False)
    commercial_mode=True
)

# Files will be created in the current working directory (existing behavior)
result = agent.go("Create some files...")
```

## Benefits of Sandbox Mode

1. **Clean Workspace**: Each session gets its own isolated directory
2. **No Clutter**: Generated files don't mix with your project files  
3. **Easy Cleanup**: Simply delete the sandbox folder when done
4. **Reproducible**: Each run starts with a clean environment
5. **Safe Exploration**: Experimental code won't affect your main workspace

## API Reference

### New Parameters

- `sandbox_mode: bool = False` - Enable/disable sandbox mode
- `sandbox_path: str | None = None` - Custom sandbox directory (optional)

### New Methods

- `agent.get_sandbox_path() -> str | None` - Get current sandbox path