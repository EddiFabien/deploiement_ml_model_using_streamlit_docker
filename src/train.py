"""Training script for water potability model."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import yaml

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load and preprocess the dataset."""
    data_path = Path(config['data']['raw'])
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path
    
    df = pd.read_csv(data_path)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    return X, y

def create_pipeline():
    """Create preprocessing and model pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        ))
    ])

def train_model(config):
    """Train the water potability prediction model."""
    # Load data
    X, y = load_data(config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    
    # Create and train pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Save model
    model_path = Path(config['model']['file_path'])
    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    # Print metrics
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    return pipeline, (X_test, y_test)

if __name__ == '__main__':
    config = load_config()
    model, (X_test, y_test) = train_model(config)
