from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from pydantic import BaseModel
from biomni_mcp_tools.get_rna_seq_archs4.tool import get_rna_seq_archs4

app = FastAPI()

class Input(BaseModel):
    gene_name: str
    K: int = 10

@app.post("/get_rna_seq_archs4")
async def run_tool(input: Input):
    result = get_rna_seq_archs4(input.gene_name, input.K)
    return {"result": result}
