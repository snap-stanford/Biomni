"""
Data Preprocessing Tools for Biomni

This module provides comprehensive data preprocessing tools for cleaning, transforming,
and preparing datasets for analysis. Includes functions for handling missing values,
outliers, normalization, encoding, and data quality assessment.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats


def load_and_inspect_data(file_path, sep=',', encoding='utf-8'):
    """
    Load dataset and provide comprehensive inspection.
    
    Args:
        file_path (str): Path to the data file
        sep (str): Separator for CSV files (default: ',')
        encoding (str): File encoding (default: 'utf-8')
    
    Returns:
        dict: Dictionary containing loaded data and inspection results
    """
    try:
        # Load data based on file extension
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path, sep=sep, encoding=encoding)
        elif file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Basic inspection
        inspection = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicates': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'unique_values': {col: data[col].nunique() for col in data.columns}
        }
        
        # Statistical summary for numeric columns
        if inspection['numeric_columns']:
            inspection['numeric_summary'] = data[inspection['numeric_columns']].describe().to_dict()
        
        print(f"✅ Successfully loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"📊 Missing values: {sum(inspection['missing_values'].values())}")
        print(f"🔄 Duplicates: {inspection['duplicates']}")
        
        return {
            'data': data,
            'inspection': inspection
        }
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def clean_missing_values(data, strategy='auto', columns=None):
    """
    Handle missing values with various strategies.
    
    Args:
        data (pd.DataFrame): Input dataset
        strategy (str): Strategy for handling missing values
                       'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'knn'
        columns (list): Specific columns to process (None for all)
    
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    data_cleaned = data.copy()
    
    if columns is None:
        columns = data.columns
    
    print(f"🧹 Cleaning missing values using strategy: {strategy}")
    
    if strategy == 'auto':
        # Automatic strategy based on data type and missing percentage
        for col in columns:
            missing_pct = data[col].isnull().sum() / len(data) * 100
            
            if missing_pct == 0:
                continue
            elif missing_pct > 50:
                print(f"⚠️  Column '{col}' has {missing_pct:.1f}% missing values - consider dropping")
                continue
            
            if data[col].dtype in ['object', 'category']:
                # Use mode for categorical
                mode_val = data[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                data_cleaned[col].fillna(fill_val, inplace=True)
            else:
                # Use median for numeric
                data_cleaned[col].fillna(data[col].median(), inplace=True)
                
    elif strategy == 'drop':
        data_cleaned = data_cleaned.dropna(subset=columns)
        
    elif strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        numeric_cols = [col for col in columns if data[col].dtype in [np.number]]
        if numeric_cols:
            data_cleaned[numeric_cols] = imputer.fit_transform(data_cleaned[numeric_cols])
            
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = [col for col in columns if data[col].dtype in [np.number]]
        if numeric_cols:
            data_cleaned[numeric_cols] = imputer.fit_transform(data_cleaned[numeric_cols])
    
    elif strategy == 'forward_fill':
        data_cleaned[columns] = data_cleaned[columns].fillna(method='ffill')
        
    elif strategy == 'backward_fill':
        data_cleaned[columns] = data_cleaned[columns].fillna(method='bfill')
    
    cleaned_missing = data_cleaned.isnull().sum().sum()
    original_missing = data.isnull().sum().sum()
    
    print(f"✅ Reduced missing values from {original_missing} to {cleaned_missing}")
    
    return data_cleaned


def remove_outliers(data, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from numeric columns.
    
    Args:
        data (pd.DataFrame): Input dataset
        columns (list): Columns to process (None for all numeric)
        method (str): Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: Dataset with outliers removed
    """
    data_clean = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    print(f"🎯 Removing outliers using {method} method")
    original_rows = len(data_clean)
    
    for col in columns:
        if col not in data.columns or data[col].dtype not in [np.number]:
            continue
            
        if method == 'iqr':
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean[col].dropna()))
            outlier_mask = pd.Series(False, index=data_clean.index)
            outlier_mask.loc[data_clean[col].dropna().index] = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = data_clean[col].median()
            mad = np.median(np.abs(data_clean[col] - median))
            if mad == 0:
                continue
            modified_z_scores = 0.6745 * (data_clean[col] - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
        
        outliers_removed = outlier_mask.sum()
        data_clean = data_clean[~outlier_mask]
        
        if outliers_removed > 0:
            print(f"  📊 Removed {outliers_removed} outliers from column '{col}'")
    
    final_rows = len(data_clean)
    print(f"✅ Dataset shape changed from {original_rows} to {final_rows} rows")
    
    return data_clean


def normalize_data(data, columns=None, method='standard'):
    """
    Normalize numeric columns.
    
    Args:
        data (pd.DataFrame): Input dataset
        columns (list): Columns to normalize (None for all numeric)
        method (str): Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
        tuple: (normalized_data, scaler_object)
    """
    data_normalized = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    print(f"📏 Normalizing data using {method} scaling")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if len(columns) > 0:
        data_normalized[columns] = scaler.fit_transform(data_normalized[columns])
        print(f"✅ Normalized {len(columns)} columns")
    
    return data_normalized, scaler


def encode_categorical_variables(data, columns=None, method='label'):
    """
    Encode categorical variables.
    
    Args:
        data (pd.DataFrame): Input dataset
        columns (list): Columns to encode (None for all categorical)
        method (str): Encoding method ('label', 'onehot', 'target')
    
    Returns:
        tuple: (encoded_data, encoder_objects)
    """
    data_encoded = data.copy()
    encoders = {}
    
    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns
    
    print(f"🏷️  Encoding categorical variables using {method} encoding")
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if method == 'label':
            encoder = LabelEncoder()
            data_encoded[col] = encoder.fit_transform(data_encoded[col].astype(str))
            encoders[col] = encoder
            
        elif method == 'onehot':
            # Create dummy variables
            dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=True)
            data_encoded = pd.concat([data_encoded.drop(col, axis=1), dummies], axis=1)
            encoders[col] = list(dummies.columns)
    
    print(f"✅ Encoded {len(columns)} categorical columns")
    
    return data_encoded, encoders


def generate_preprocessing_report(original_data, processed_data, steps_performed):
    """
    Generate a comprehensive preprocessing report.
    
    Args:
        original_data (pd.DataFrame): Original dataset
        processed_data (pd.DataFrame): Processed dataset
        steps_performed (list): List of preprocessing steps performed
    
    Returns:
        dict: Preprocessing report
    """
    report = {
        'original_shape': original_data.shape,
        'processed_shape': processed_data.shape,
        'steps_performed': steps_performed,
        'shape_change': {
            'rows_change': processed_data.shape[0] - original_data.shape[0],
            'columns_change': processed_data.shape[1] - original_data.shape[1]
        },
        'missing_values': {
            'original': original_data.isnull().sum().sum(),
            'processed': processed_data.isnull().sum().sum()
        },
        'data_types': {
            'original': original_data.dtypes.value_counts().to_dict(),
            'processed': processed_data.dtypes.value_counts().to_dict()
        }
    }
    
    print("\n" + "="*50)
    print("📋 PREPROCESSING REPORT")
    print("="*50)
    print(f"Original shape: {report['original_shape']}")
    print(f"Processed shape: {report['processed_shape']}")
    print(f"Rows change: {report['shape_change']['rows_change']}")
    print(f"Columns change: {report['shape_change']['columns_change']}")
    print(f"Missing values: {report['missing_values']['original']} → {report['missing_values']['processed']}")
    print(f"Steps performed: {', '.join(steps_performed)}")
    print("="*50)
    
    return report


def detect_data_quality_issues(data, missing_threshold=0.5, high_cardinality_threshold=100, correlation_threshold=0.95):
    """
    Detect common data quality issues in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataset
        missing_threshold (float): Threshold for flagging high missing value columns
        high_cardinality_threshold (int): Threshold for flagging high cardinality categorical variables
        correlation_threshold (float): Threshold for flagging highly correlated features
    
    Returns:
        dict: Dictionary containing detected issues
    """
    issues = {
        'high_missing_columns': [],
        'high_cardinality_categorical': [],
        'highly_correlated_pairs': [],
        'potential_duplicates': 0,
        'data_type_issues': [],
        'outlier_columns': []
    }
    
    print("🔍 Detecting data quality issues...")
    
    # High missing value columns
    missing_pct = data.isnull().sum() / len(data)
    high_missing = missing_pct[missing_pct > missing_threshold]
    issues['high_missing_columns'] = high_missing.to_dict()
    
    # High cardinality categorical variables
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = data[col].nunique()
        if unique_count > high_cardinality_threshold:
            issues['high_cardinality_categorical'].append({
                'column': col,
                'unique_count': unique_count
            })
    
    # Highly correlated features
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for col in upper_triangle.columns:
            for row in upper_triangle.index:
                if upper_triangle.loc[row, col] > correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': row,
                        'feature2': col,
                        'correlation': upper_triangle.loc[row, col]
                    })
        issues['highly_correlated_pairs'] = high_corr_pairs
    
    # Potential duplicates
    issues['potential_duplicates'] = data.duplicated().sum()
    
    # Data type issues (numeric columns stored as objects)
    for col in data.select_dtypes(include=['object']).columns:
        try:
            pd.to_numeric(data[col], errors='raise')
            issues['data_type_issues'].append({
                'column': col,
                'issue': 'Numeric data stored as object'
            })
        except (ValueError, TypeError):
            pass
    
    # Outlier detection for numeric columns
    for col in numeric_data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > 0:
            issues['outlier_columns'].append({
                'column': col,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100
            })
    
    # Print summary
    print(f"📊 Found {len(issues['high_missing_columns'])} high missing value columns")
    print(f"🔢 Found {len(issues['high_cardinality_categorical'])} high cardinality categorical variables")
    print(f"🔗 Found {len(issues['highly_correlated_pairs'])} highly correlated feature pairs")
    print(f"👥 Found {issues['potential_duplicates']} potential duplicate rows")
    print(f"🏷️  Found {len(issues['data_type_issues'])} data type issues")
    print(f"📈 Found {len(issues['outlier_columns'])} columns with outliers")
    
    return issues


def create_data_preprocessing_pipeline(data, missing_strategy='auto', outlier_method='iqr', 
                                     normalization_method='standard', encoding_method='label',
                                     save_path=None):
    """
    Create a complete data preprocessing pipeline.
    
    Args:
        data (pd.DataFrame): Input dataset
        missing_strategy (str): Strategy for handling missing values
        outlier_method (str): Method for outlier detection and removal
        normalization_method (str): Method for feature normalization
        encoding_method (str): Method for categorical encoding
        save_path (str): Path to save the processed data (optional)
    
    Returns:
        dict: Dictionary containing processed data and pipeline components
    """
    print("🔄 Starting complete preprocessing pipeline...")
    
    original_data = data.copy()
    processed_data = data.copy()
    pipeline_steps = []
    encoders = {}
    scalers = {}
    
    # Step 1: Handle missing values
    print("\n📍 Step 1: Handling missing values")
    processed_data = clean_missing_values(processed_data, strategy=missing_strategy)
    pipeline_steps.append(f"Missing values handled with {missing_strategy} strategy")
    
    # Step 2: Remove outliers
    print("\n📍 Step 2: Removing outliers")
    processed_data = remove_outliers(processed_data, method=outlier_method)
    pipeline_steps.append(f"Outliers removed using {outlier_method} method")
    
    # Step 3: Encode categorical variables
    categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\n📍 Step 3: Encoding categorical variables")
        processed_data, encoders = encode_categorical_variables(
            processed_data, columns=categorical_cols, method=encoding_method
        )
        pipeline_steps.append(f"Categorical variables encoded using {encoding_method} encoding")
    
    # Step 4: Normalize numeric features
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\n📍 Step 4: Normalizing numeric features")
        processed_data, scaler = normalize_data(
            processed_data, columns=numeric_cols, method=normalization_method
        )
        scalers['numeric_scaler'] = scaler
        pipeline_steps.append(f"Numeric features normalized using {normalization_method} scaling")
    
    # Step 5: Generate report
    print("\n📍 Step 5: Generating preprocessing report")
    report = generate_preprocessing_report(original_data, processed_data, pipeline_steps)
    
    # Save processed data if path provided
    if save_path:
        processed_data.to_csv(save_path, index=False)
        print(f"💾 Processed data saved to: {save_path}")
    
    pipeline_result = {
        'original_data': original_data,
        'processed_data': processed_data,
        'encoders': encoders,
        'scalers': scalers,
        'pipeline_steps': pipeline_steps,
        'report': report
    }
    
    print("\n✅ Preprocessing pipeline completed successfully!")
    
    return pipeline_result