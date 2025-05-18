"""Test cases for utility functions."""
import pytest
import pandas as pd
import numpy as np
from src.utils import validate_input, create_input_dataframe


def test_validate_input_valid_number():
    """Test validate_input with valid number."""
    assert validate_input("42.5") == 42.5


def test_validate_input_invalid_input():
    """Test validate_input with invalid input."""
    assert np.isnan(validate_input("invalid"))


def test_validate_input_empty_string():
    """Test validate_input with empty string."""
    assert np.isnan(validate_input(""))


def test_create_input_dataframe():
    """Test create_input_dataframe with valid inputs."""
    test_inputs = {
        "ph": "7.5",
        "Hardness": "200.0",
        "Solids": "20000"
    }
    df = create_input_dataframe(test_inputs)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["ph"].iloc[0] == 7.5
    assert df["Hardness"].iloc[0] == 200.0
    assert df["Solids"].iloc[0] == 20000.0


def test_create_input_dataframe_with_invalid():
    """Test create_input_dataframe with some invalid inputs."""
    test_inputs = {
        "ph": "invalid",
        "Hardness": "200.0"
    }
    df = create_input_dataframe(test_inputs)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert np.isnan(df["ph"].iloc[0])
    assert df["Hardness"].iloc[0] == 200.0
