"""Test cases for model wrapper."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.model import WaterPotabilityModel


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.predict.return_value = np.array([1])
    mock.predict_proba.return_value = np.array([[0.3, 0.7]])
    return mock


@pytest.fixture
def test_input():
    """Create test input data."""
    return pd.DataFrame({
        'ph': [7.5],
        'Hardness': [200.0],
        'Solids': [20000.0],
        'Chloramines': [7.0],
        'Sulfate': [350.0],
        'Conductivity': [360.0],
        'Organic_carbon': [18.0],
        'Trihalomethanes': [100.0],
        'Turbidity': [4.0]
    })


def test_model_prediction(mock_model, test_input):
    """Test model prediction with mock model."""
    with patch('joblib.load', return_value=mock_model):
        model = WaterPotabilityModel()
        prediction, probability = model.predict(test_input)
        
        assert prediction == "Potable"
        assert probability == 70.0  # 0.7 * 100


def test_model_prediction_no_proba(test_input):
    """Test model prediction when predict_proba is not available."""
    mock = Mock()
    mock.predict.return_value = np.array([0])
    
    with patch('joblib.load', return_value=mock):
        model = WaterPotabilityModel()
        prediction, probability = model.predict(test_input)
        
        assert prediction == "Not Potable"
        assert probability is None


def test_model_prediction_error(test_input):
    """Test model prediction when an error occurs."""
    mock = Mock()
    mock.predict.side_effect = Exception("Test error")
    
    with patch('joblib.load', return_value=mock):
        model = WaterPotabilityModel()
        prediction, probability = model.predict(test_input)
        
        assert prediction == "Unknown"
        assert probability is None
