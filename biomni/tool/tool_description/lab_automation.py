description = [
    {
        "description": "Generate a PyLabRobot liquid handling script based on natural language description of liquid handling tasks.",
        "name": "pylabrobot_liquid_handling_script",
        "optional_parameters": [
            {
                "default": None,
                "description": "Alternative parameter name for the liquid handling task description",
                "name": "task_description",
                "type": "str",
            },
            {
                "default": True,
                "description": "If True, return full response including raw LLM output",
                "name": "verbose",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Directory to save the generated script. If provided, the script will be saved as a .py file in this directory",
                "name": "save_to_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language description of the liquid handling task (e.g., 'Move 100μL from wells A1:A3 to B1:B3')",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Generate a PyLabRobot material handling script based on natural language description of material handling tasks using the iSWAP gripper module.",
        "name": "pylabrobot_material_handling_script",
        "optional_parameters": [
            {
                "default": None,
                "description": "Alternative parameter name for the material handling task description",
                "name": "task_description",
                "type": "str",
            },
            {
                "default": True,
                "description": "If True, return full response including raw LLM output",
                "name": "verbose",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Directory to save the generated script. If provided, the script will be saved as a .py file in this directory",
                "name": "save_to_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language description of the material handling task (e.g., 'Move plate from position A to position B')",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Test a PyLabRobot script based on the script content.",
        "name": "test_pylabrobot_script",
        "optional_parameters": [
            {
                "default": False,
                "description": "If True, enable tracking of the script execution",
                "name": "enable_tracking",
                "type": "bool",
            },
            {
                "default": 60,
                "description": "Timeout in seconds for the script execution",
                "name": "timeout_seconds",
                "type": "int",
            },
            {
                "default": False,
                "description": "If True, save the test results as a .json file",
                "name": "save_test_report",
                "type": "bool",
            },
            {
                "default": None,
                "description": "Directory to save the test results. If provided, the test results will be saved as a .json file in this directory",
                "name": "test_report_dir",
                "type": "str",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Script content to test",
                "name": "script_input",
                "type": "str",
            }
        ],
    },
]
