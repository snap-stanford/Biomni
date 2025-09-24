# generate_api_docs.py

import os


def generate_api_docs(base_path, output_dir, package_name, exclude_list=None):
    """
    Automatically generates .rst files for Sphinx API documentation.

    Args:
        base_path (str): The root directory of the Python package (e.g., 'biomni').
        output_dir (str): The directory where the .rst files will be generated (e.g., 'source/api').
        package_name (str): The name of the Python package.
        exclude_list (list): A list of paths (directories or files) to exclude from documentation.
                             Paths should be relative to the base_path.
    """
    if exclude_list is None:
        exclude_list = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the package directory
    for root, dirs, files in os.walk(base_path):
        # Exclude directories if they are in the exclude list
        dirs[:] = [d for d in dirs if os.path.join(os.path.relpath(root, base_path), d) not in exclude_list]

        for file in files:
            file_path_relative = os.path.relpath(os.path.join(root, file), base_path)

            # Check if the file should be excluded
            if file_path_relative in exclude_list:
                print(f"Skipping excluded file: {file_path_relative}")
                continue

            if file.endswith(".py") and file != "__init__.py":
                # Construct the full module name (e.g., biomni.agent.a1)
                rel_path = os.path.relpath(root, base_path)
                module_name = f"{package_name}.{rel_path.replace(os.path.sep, '.')}.{os.path.splitext(file)[0]}"

                # Clean up the module name if it starts with the package name
                if module_name.startswith(f"{package_name}.{package_name}."):
                    module_name = f"{package_name}.{module_name[len(f'{package_name}.{package_name}.') :]}"

                # Create the directory structure in the output folder
                output_path_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(output_path_dir):
                    os.makedirs(output_path_dir)

                # Define the output .rst file path
                output_file = os.path.join(output_path_dir, f"{os.path.splitext(file)[0]}.rst")

                # Generate the RST content
                rst_content = f"""{module_name}
{"=" * len(module_name)}

.. automodule:: {module_name}
   :members:
   :undoc-members:
   :show-inheritance:
"""
                # Write the content to the .rst file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(rst_content)
                print(f"Generated {output_file}")


def create_index_rst(output_dir, package_name):
    """
    Creates or updates the main index file that lists all generated .rst files.
    """
    output_dir = os.path.join(output_dir, "../")
    index_path = os.path.join(output_dir, "index.rst")
    rst_files = []

    # Find all generated rst files
    for root, _dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".rst") and file != "index.rst":
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                rst_files.append(os.path.splitext(rel_path)[0])

    rst_files.sort()

    index_content = f"""{package_name} API Reference
{"=" * (len(package_name) + len(" API Reference"))}

.. toctree::
   :maxdepth: 2
   :caption: API Contents:

"""
    for file in rst_files:
        index_content += f"   {file}\n"

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)
    print(f"Updated {index_path} with module links.")

    # update ../index.rst to include api/index
    # index_path = os.path.join(output_dir, '../index.rst')
    # with open(index_path, 'w', encoding='utf-8') as f:


if __name__ == "__main__":
    package_to_document = "../biomni"
    api_docs_path = "./source/api"
    package_name = "biomni"

    os.makedirs(api_docs_path, exist_ok=True)

    exclude_list = [
        "llm.py",
        "env_desc.py",
        "version.py",
        "tool/tool_description",
        "tool/example_mcp_tools",
    ]

    generate_api_docs(package_to_document, api_docs_path, package_name, exclude_list)
    create_index_rst(api_docs_path, package_name)

    print("\nAPI documentation source files have been generated.")
    print("Now run 'make html' from your project's root directory to build the documentation.")
