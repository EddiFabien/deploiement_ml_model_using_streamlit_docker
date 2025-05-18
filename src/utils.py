"""Utility functions for water potability prediction."""
import pandas as pd
from typing import Union, Dict


def validate_input(input_value: str) -> float:
    """Convert and validate input string to float.
    
    Args:
        input_value: String value to convert
        
    Returns:
        float: Converted value or NaN if invalid
    """
    try:
        return float(input_value)
    except (ValueError, TypeError):
        return float('NaN')


def create_input_dataframe(inputs: Dict[str, str]) -> pd.DataFrame:
    """Create a DataFrame from input values.
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        pd.DataFrame: DataFrame ready for model prediction
    """
    validated_inputs = {
        key: validate_input(value)
        for key, value in inputs.items()
    }
    return pd.DataFrame([validated_inputs])
