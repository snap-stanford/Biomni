description = [
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
    {
        "description": "Get the documentation for the liquid handling section of the PyLabRobot tutorial.",
        "name": "get_pylabrobot_documentation_liquid",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": "Get the documentation for the material handling section of the PyLabRobot tutorial.",
        "name": "get_pylabrobot_documentation_material",
        "optional_parameters": [],
        "required_parameters": [],
    },
]
