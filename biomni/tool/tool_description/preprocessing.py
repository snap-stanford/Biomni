"""
Tool descriptions for data preprocessing functions.

This module contains the API descriptions for all preprocessing tools,
formatted for the Biomni agent system.
"""

preprocessing_tools = [
    {
        "name": "load_and_inspect_data",
        "description": "Load dataset from various formats (CSV, Excel, JSON, Parquet) and provide comprehensive inspection including shape, data types, missing values, duplicates, and statistical summaries.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [{"name": "file_path", "type": "str", "description": "Path to the data file to load"}],
        "optional_parameters": [
            {"name": "sep", "type": "str", "description": "Separator for CSV files", "default": ","},
            {"name": "encoding", "type": "str", "description": "File encoding", "default": "utf-8"},
        ],
        "return_type": "dict",
        "return_description": "Dictionary containing loaded data and comprehensive inspection results",
    },
    {
        "name": "clean_missing_values",
        "description": "Handle missing values in dataset using various strategies including automatic detection, statistical imputation, KNN imputation, forward/backward fill, or row dropping.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [{"name": "data", "type": "pd.DataFrame", "description": "Input dataset to clean"}],
        "optional_parameters": [
            {
                "name": "strategy",
                "type": "str",
                "description": "Strategy for handling missing values: 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'knn'",
                "default": "auto",
            },
            {
                "name": "columns",
                "type": "list",
                "description": "Specific columns to process (None for all columns)",
                "default": None,
            },
        ],
        "return_type": "pd.DataFrame",
        "return_description": "Dataset with missing values handled according to specified strategy",
    },
    {
        "name": "remove_outliers",
        "description": "Detect and remove outliers from numeric columns using IQR, Z-score, or modified Z-score methods with configurable thresholds.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [
            {"name": "data", "type": "pd.DataFrame", "description": "Input dataset for outlier removal"}
        ],
        "optional_parameters": [
            {
                "name": "columns",
                "type": "list",
                "description": "Columns to process (None for all numeric columns)",
                "default": None,
            },
            {
                "name": "method",
                "type": "str",
                "description": "Outlier detection method: 'iqr', 'zscore', 'modified_zscore'",
                "default": "iqr",
            },
            {"name": "threshold", "type": "float", "description": "Threshold for outlier detection", "default": 1.5},
        ],
        "return_type": "pd.DataFrame",
        "return_description": "Dataset with outliers removed from specified columns",
    },
    {
        "name": "normalize_data",
        "description": "Normalize numeric features using StandardScaler, MinMaxScaler, or RobustScaler to prepare data for machine learning algorithms.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [{"name": "data", "type": "pd.DataFrame", "description": "Input dataset to normalize"}],
        "optional_parameters": [
            {
                "name": "columns",
                "type": "list",
                "description": "Columns to normalize (None for all numeric columns)",
                "default": None,
            },
            {
                "name": "method",
                "type": "str",
                "description": "Normalization method: 'standard', 'minmax', 'robust'",
                "default": "standard",
            },
        ],
        "return_type": "tuple",
        "return_description": "Tuple containing (normalized_data, scaler_object) for future transformations",
    },
    {
        "name": "encode_categorical_variables",
        "description": "Encode categorical variables using label encoding or one-hot encoding to convert text categories into numeric format suitable for analysis.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [
            {
                "name": "data",
                "type": "pd.DataFrame",
                "description": "Input dataset with categorical variables to encode",
            }
        ],
        "optional_parameters": [
            {
                "name": "columns",
                "type": "list",
                "description": "Columns to encode (None for all categorical columns)",
                "default": None,
            },
            {"name": "method", "type": "str", "description": "Encoding method: 'label', 'onehot'", "default": "label"},
        ],
        "return_type": "tuple",
        "return_description": "Tuple containing (encoded_data, encoder_objects) for future transformations",
    },
    {
        "name": "generate_preprocessing_report",
        "description": "Generate comprehensive report comparing original and processed datasets, showing changes in shape, missing values, data types, and preprocessing steps performed.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [
            {"name": "original_data", "type": "pd.DataFrame", "description": "Original dataset before preprocessing"},
            {"name": "processed_data", "type": "pd.DataFrame", "description": "Dataset after preprocessing"},
            {
                "name": "steps_performed",
                "type": "list",
                "description": "List of preprocessing steps that were performed",
            },
        ],
        "optional_parameters": [],
        "return_type": "dict",
        "return_description": "Comprehensive preprocessing report with before/after comparisons",
    },
    {
        "name": "detect_data_quality_issues",
        "description": "Automatically detect common data quality issues including high missing values, high cardinality categoricals, highly correlated features, duplicates, data type problems, and outliers.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [
            {"name": "data", "type": "pd.DataFrame", "description": "Dataset to analyze for quality issues"}
        ],
        "optional_parameters": [
            {
                "name": "missing_threshold",
                "type": "float",
                "description": "Threshold for flagging high missing value columns",
                "default": 0.5,
            },
            {
                "name": "high_cardinality_threshold",
                "type": "int",
                "description": "Threshold for flagging high cardinality categorical variables",
                "default": 100,
            },
            {
                "name": "correlation_threshold",
                "type": "float",
                "description": "Threshold for flagging highly correlated features",
                "default": 0.95,
            },
        ],
        "return_type": "dict",
        "return_description": "Dictionary containing detected data quality issues and recommendations",
    },
    {
        "name": "create_data_preprocessing_pipeline",
        "description": "Execute complete end-to-end preprocessing pipeline including missing value handling, outlier removal, categorical encoding, feature normalization, and reporting with optional data saving.",
        "module": "biomni.tool.preprocessing",
        "required_parameters": [{"name": "data", "type": "pd.DataFrame", "description": "Input dataset to preprocess"}],
        "optional_parameters": [
            {
                "name": "missing_strategy",
                "type": "str",
                "description": "Strategy for handling missing values",
                "default": "auto",
            },
            {
                "name": "outlier_method",
                "type": "str",
                "description": "Method for outlier detection and removal",
                "default": "iqr",
            },
            {
                "name": "normalization_method",
                "type": "str",
                "description": "Method for feature normalization",
                "default": "standard",
            },
            {
                "name": "encoding_method",
                "type": "str",
                "description": "Method for categorical encoding",
                "default": "label",
            },
            {
                "name": "save_path",
                "type": "str",
                "description": "Path to save the processed data (optional)",
                "default": None,
            },
        ],
        "return_type": "dict",
        "return_description": "Dictionary containing processed data, encoders, scalers, pipeline steps, and comprehensive report",
    },
]
