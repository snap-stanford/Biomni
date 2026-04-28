import pickle

import pandas as pd


class ToolRegistry:
    def __init__(self, tools):
        self.tools = []
        self.next_id = 0

        for j in tools.values():
            for tool in j:
                self.register_tool(tool)

        docs = []
        for tool_id in range(len(self.tools)):
            docs.append([int(tool_id), self.get_tool_by_id(int(tool_id))])
        self.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])

        # self.langchain_tools = {}
        # for module, api_list in tools.items():
        #    self.langchain_tools.update({self.get_id_by_name(api['name']): api_schema_to_langchain_tool(api, mode = 'custom_tool', module_name = module) for api in api_list})

    def register_tool(self, tool):
        if self.validate_tool(tool):
            tool["id"] = self.next_id
            self.tools.append(tool)
            self.next_id += 1
        else:
            raise ValueError("Invalid tool format")

    def validate_tool(self, tool):
        required_keys = ["name", "description", "required_parameters"]
        return all(key in tool for key in required_keys)

    def get_tool_by_name(self, name):
        for tool in self.tools:
            if tool["name"] == name:
                return tool
        return None

    def get_tool_by_id(self, tool_id):
        for tool in self.tools:
            if tool["id"] == tool_id:
                return tool
        return None

    def get_id_by_name(self, name):
        for tool in self.tools:
            if tool["name"] == name:
                return tool["id"]
        return None

    def get_name_by_id(self, tool_id):
        for tool in self.tools:
            if tool["id"] == tool_id:
                return tool["name"]
        return None

    def list_tools(self):
        return [{"name": tool["name"], "id": tool["id"]} for tool in self.tools]

    def remove_tool_by_id(self, tool_id):
        # Remove the tool with the given id
        tool = self.get_tool_by_id(tool_id)
        if tool:
            self.tools = [t for t in self.tools if t["id"] != tool_id]
            return True
        return False

    def remove_tool_by_name(self, name):
        # Remove the tool with the given name
        tool = self.get_tool_by_name(name)
        if tool:
            self.tools = [t for t in self.tools if t["name"] != name]
            return True
        return False

    def save_registry(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    # def get_langchain_tool_by_id(self, id):
    #     return self.langchain_tools[id]

    @staticmethod
    def load_registry(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
