from biomni.env_desc import data_lake_dict, library_content_dict
from biomni.utils import read_module2api
import glob
from biomni.tool.tool_registry import ToolRegistry
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document

data_path = "/workdir_efs/jaechang/work2/biomni_hits_test/biomni_data/biomni_data"

embeddings = BedrockEmbeddings(normalize=True)

# Prepare tool descriptions
module2api = read_module2api()

tool_registry = ToolRegistry(module2api)
all_tools = tool_registry.tools

data_lake_path = data_path + "/data_lake"
data_lake_content = glob.glob(data_lake_path + "/*")
data_lake_items = [x.split("/")[-1] for x in data_lake_content]

# Create data lake descriptions for retrieval
data_lake_descriptions = []
for item in data_lake_items:
    description = data_lake_dict.get(item, f"Data lake item: {item}")
    data_lake_descriptions.append({"name": item, "description": description})

# 3. Libraries with descriptions - use library_content_dict directly
library_descriptions = []
for lib_name, lib_desc in library_content_dict.items():
    library_descriptions.append({"name": lib_name, "description": lib_desc})


def save_db(name_and_description_list, db_name):
    texts = []
    for tool in name_and_description_list:
        name = tool["name"]
        description = tool["description"]
        texts.append(
            Document(
                page_content=f"{name}: {description}",
                metadata=tool,
            )
        )

    db = FAISS.from_documents(
        texts,
        embeddings,
        normalize_L2=True,
        # distance_strategy=DistanceStrategy.COSINE,
    )
    db.save_local(db_name)


save_db(all_tools, "tool_index")
save_db(data_lake_descriptions, "data_lake_index")
save_db(library_descriptions, "library_index")
