from biomni_mcp_tools.genetics.dynamic_loader import ToolLoader
from fastapi import FastAPI
from pydantic import BaseModel, create_model

app = FastAPI()
loader = ToolLoader(config_path="/Users/shinde/Desktop/Biomni/biomni_mcp_tools/genetics/tools_config.yaml")


# Dynamically create endpoints for each tool
def register_tool_endpoint(tool_name, tool_info):
    input_fields = {k: (eval(v), ...) for k, v in tool_info["input_schema"].items()}
    InputModel = create_model(f"{tool_name}Input", **input_fields)

    @app.post(tool_info["endpoint"])
    def endpoint(input: InputModel):
        result = tool_info["func"](**input.dict())
        return {"result": result}


for name, info in loader.tools.items():
    register_tool_endpoint(name, info)
