"""
Utility functions for data validation and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def validate_time_series(data: pd.DataFrame, 
                         required_columns: list,
                         date_column: str = 'date') -> Tuple[bool, str]:
    """
    Validate time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to validate
    required_columns : list
        List of required column names
    date_column : str
        Name of the date/time column
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if data is None or len(data) == 0:
        return False, "Data is empty"
    
    # Check required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check date column
    if date_column in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            return False, f"{date_column} must be datetime type"
    
    return True, "Valid"


def handle_missing_values(data: pd.DataFrame, 
                          method: str = 'interpolate',
                          limit: Optional[int] = None) -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with potential missing values
    method : str
        Method to handle missing values: 'interpolate', 'ffill', 'bfill', 'drop'
    limit : int, optional
        Maximum number of consecutive NaNs to fill
        
    Returns:
    --------
    pd.DataFrame
        Data with missing values handled
    """
    if method == 'interpolate':
        return data.interpolate(method='time', limit=limit)
    elif method == 'ffill':
        return data.fillna(method='ffill', limit=limit)
    elif method == 'bfill':
        return data.fillna(method='bfill', limit=limit)
    elif method == 'drop':
        return data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_features(data: pd.DataFrame, 
                      method: str = 'zscore') -> pd.DataFrame:
    """
    Normalize features for model input.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Features to normalize
    method : str
        Normalization method: 'zscore', 'minmax', 'robust'
        
    Returns:
    --------
    pd.DataFrame
        Normalized features
    """
    if method == 'zscore':
        return (data - data.mean()) / data.std()
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'robust':
        median = data.median()
        iqr = data.quantile(0.75) - data.quantile(0.25)
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_seasonality(data: pd.Series, 
                         period: int = 12) -> pd.Series:
    """
    Calculate seasonal component of a time series.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    period : int
        Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        
    Returns:
    --------
    pd.Series
        Seasonal component
    """
    # Simple seasonal decomposition
    seasonal = data.rolling(window=period, center=True).mean()
    return seasonal


def detect_outliers(data: pd.Series, 
                   method: str = 'iqr',
                   threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in time series data.
    
    Parameters:
    -----------
    data : pd.Series
        Data to check for outliers
    method : str
        Detection method: 'iqr', 'zscore'
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    else:
        raise ValueError(f"Unknown method: {method}")


def align_time_series(series_list: list, 
                     method: str = 'inner') -> list:
    """
    Align multiple time series to common time index.
    
    Parameters:
    -----------
    series_list : list
        List of pd.Series or pd.DataFrame with datetime index
    method : str
        Alignment method: 'inner', 'outer', 'left', 'right'
        
    Returns:
    --------
    list
        List of aligned time series
    """
    if not series_list:
        return []
    
    # Find common index based on method
    if method == 'inner':
        common_idx = series_list[0].index
        for series in series_list[1:]:
            common_idx = common_idx.intersection(series.index)
    elif method == 'outer':
        common_idx = series_list[0].index
        for series in series_list[1:]:
            common_idx = common_idx.union(series.index)
    else:
        common_idx = series_list[0].index
    
    # Reindex all series
    aligned = [series.reindex(common_idx) for series in series_list]
    
    return aligned
