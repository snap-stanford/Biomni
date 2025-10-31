# Source Generated with Decompyle++
# File: omics_horizon_config.cpython-312.pyc (Python 3.12)

'''
OmicsHorizon Configuration Module

Contains all configuration constants and settings for the OmicsHorizon application.
'''
LLM_MODEL = 'gemini-2.5-pro'
BIOMNI_DATA_PATH = '/workdir_efs/jhjeon/Biomni/biomni_data'
WORK_DIR = '/workdir_efs/jhjeon/Biomni/streamlit_workspace'
LOGO_COLOR_PATH = 'logo/OMICS-HORIZON_Logo_Color.svg'
LOGO_MONO_PATH = 'logo/OMICS-HORIZON_Logo_Mono.svg'
MAX_DATA_COLUMNS_TO_SHOW = 20
MAX_SAMPLE_EXAMPLES = 5
MIN_COLUMN_PATTERN_LENGTH = 3
MAX_CONTENT_LENGTH_FOR_LLM = 15000
MAX_DISPLAY_TEXT_LENGTH = 8000
MAX_OBSERVATION_DISPLAY_LENGTH = 2000
MIN_MEANINGFUL_CONTENT_LENGTH = 50
APP_CSS = '\n<style>\n    .main-header {\n        font-size: 2.5rem;\n        font-weight: bold;\n        color: #1f77b4;\n        text-align: center;\n        margin-bottom: 1rem;\n    }\n    .panel-header {\n        font-size: 1.5rem;\n        font-weight: bold;\n        color: #2c3e50;\n        padding: 10px;\n        background-color: #f0f2f6;\n        border-radius: 5px;\n        margin-bottom: 1rem;\n    }\n    .stTextArea textarea {\n        font-family: monospace;\n    }\n    div[data-testid="stExpander"] {\n        border: 2px solid #e0e0e0;\n        border-radius: 10px;\n    }\n    \n    /* Wider sidebar for Q&A */\n    [data-testid="stSidebar"] {\n        min-width: 420px !important;\n        max-width: 420px !important;\n    }\n    \n    /* Logo container styling */\n    .logo-container {\n        display: flex;\n        justify-content: center;\n        align-items: center;\n        padding: 1rem 0;\n        margin-bottom: -2rem;\n    }\n    .logo-container img {\n        max-width: 100%;\n        height: auto;\n    }\n    \n    /* Light mode: show color logo, hide mono logo */\n    .logo-light {\n        display: block !important;\n    }\n    .logo-dark {\n        display: none !important;\n    }\n    \n    /* Dark mode: hide color logo, show mono logo */\n    @media (prefers-color-scheme: dark) {\n        .logo-light {\n            display: none !important;\n        }\n        .logo-dark {\n            display: block !important;\n        }\n    }\n    \n    /* Streamlit theme detection */\n    [data-theme="dark"] .logo-light,\n    [data-baseweb-theme="dark"] .logo-light,\n    .stApp[data-theme="dark"] .logo-light {\n        display: none !important;\n    }\n    \n    [data-theme="dark"] .logo-dark,\n    [data-baseweb-theme="dark"] .logo-dark,\n    .stApp[data-theme="dark"] .logo-dark {\n        display: block !important;\n    }\n    \n    /* Main page logo styling */\n    .main-logo {\n        margin: 0 auto;\n        position: relative;\n    }\n    \n    /* Ensure proper z-index stacking */\n    .logo-light {\n        z-index: 2;\n    }\n    .logo-dark {\n        z-index: 1;\n    }\n</style>\n'
