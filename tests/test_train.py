"""Test cases for training script."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
from src.train import load_config, load_data, create_pipeline, train_model

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return {
        'data': {
            'raw': 'data/raw/water_potability.csv'
        },
        'model': {
            'file_path': 'models/potability_model.pkl'
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        }
    }

@pytest.fixture
def mock_data():
    """Create mock training data."""
    # Create more realistic test data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'ph': np.random.normal(7.0, 1.0, n_samples),
        'Hardness': np.random.normal(200.0, 50.0, n_samples),
        'Solids': np.random.normal(20000.0, 5000.0, n_samples),
        'Chloramines': np.random.normal(7.0, 2.0, n_samples),
        'Sulfate': np.random.normal(350.0, 100.0, n_samples),
        'Conductivity': np.random.normal(400.0, 100.0, n_samples),
        'Organic_carbon': np.random.normal(15.0, 5.0, n_samples),
        'Trihalomethanes': np.random.normal(70.0, 20.0, n_samples),
        'Turbidity': np.random.normal(4.0, 1.0, n_samples)
    })
    
    # Generate target variable based on some rules
    y = pd.Series([
        1 if (row.ph >= 6.5 and row.ph <= 8.5 and
              row.Turbidity < 5.0 and
              row.Solids < 25000.0)
        else 0
        for _, row in X.iterrows()
    ])
    
    return X, y

def test_load_config(tmp_path):
    """Test loading configuration from YAML file."""
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    config_file = config_dir / 'model_config.yaml'
    
    config_content = """data:
  raw: data/raw/water_potability.csv
model:
  file_path: models/potability_model.pkl
training:
  test_size: 0.2
  random_state: 42"""
    config_file.write_text(config_content)
    
    with patch('src.train.Path') as mock_path:
        mock_path.return_value = config_file
        config = load_config()
        assert config['data']['raw'] == 'data/raw/water_potability.csv'
        assert config['model']['file_path'] == 'models/potability_model.pkl'
        assert config['training']['test_size'] == 0.2
        assert config['training']['random_state'] == 42

def test_load_data(mock_config):
    """Test loading and preprocessing data."""
    mock_df = pd.DataFrame({
        'Potability': [1, 0, 1],
        'ph': [7.5, 7.2, 6.8],
        'Hardness': [200.0, 180.0, 220.0]
    })
    
    with patch('pandas.read_csv', return_value=mock_df):
        X, y = load_data(mock_config)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'Potability' not in X.columns
        assert len(X) == len(y)

def test_create_pipeline():
    """Test creation of preprocessing and model pipeline."""
    pipeline = create_pipeline()
    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == 'imputer'
    assert pipeline.steps[1][0] == 'scaler'
    assert pipeline.steps[2][0] == 'classifier'

def test_train_model(mock_config, mock_data, tmp_path):
    """Test model training and saving."""
    X, y = mock_data
    model_dir = tmp_path / 'models'
    model_dir.mkdir()
    
    mock_config['model']['file_path'] = str(model_dir / 'potability_model.pkl')
    
    with patch('joblib.dump') as mock_dump:
        pipeline, (X_test, y_test) = train_model(mock_config)
        
        # Check if model was trained
        assert hasattr(pipeline, 'predict')
        assert hasattr(pipeline, 'predict_proba')
        
        # Check if model was saved
        mock_dump.assert_called_once()
        
        # Check if data was split correctly
        assert len(X_test) < len(X)
        assert len(y_test) < len(y)
