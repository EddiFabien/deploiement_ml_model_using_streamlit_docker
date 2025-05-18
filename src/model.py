"""Model handling for water potability prediction."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List

class WaterPotabilityModel:
    """A class to handle water potability predictions using a trained model.
    
    Attributes:
        model: The loaded machine learning model
        feature_names: List of feature names expected by the model
    """
    
    def __init__(self, model_path: str = '../models/potability_model.pkl'):
        """Initialize the model.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / model_path
            self.model = joblib.load(model_path)
            self.feature_names = [
                'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")
    
    def validate_features(self, X: pd.DataFrame) -> None:
        """Validate that input features match expected features.
        
        Args:
            X: Input features DataFrame
            
        Raises:
            ValueError: If features don't match expected features
        """
        if not all(feature in X.columns for feature in self.feature_names):
            missing = [f for f in self.feature_names if f not in X.columns]
            raise ValueError(f"Missing features: {missing}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[str, Union[float, None]]:
        """Make a prediction for water potability.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            tuple: (prediction string, confidence percentage)
            
        Raises:
            Exception: If there's an error during prediction
        """
        try:
            self.validate_features(X)
            X = X[self.feature_names]  # Ensure correct feature order
            prediction = self.model.predict(X)[0]
            potability = "Potable" if prediction == 1 else "Not Potable"
            
            # Get probability if available
            probability = None
            if hasattr(self.model, "predict_proba"):
                try:
                    probability = self.model.predict_proba(X)[0][1] * 100
                except Exception:
                    pass
            
            return potability, probability
        except Exception as e:
            return "Unknown", None
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates for predictions.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            numpy.ndarray: Array of prediction probabilities
            
        Raises:
            Exception: If there's an error getting probabilities
        """
        try:
            self.validate_features(X)
            X = X[self.feature_names]  # Ensure correct feature order
            return self.model.predict_proba(X)
        except Exception as e:
            raise Exception(f"Error getting prediction probabilities: {str(e)}")
    
    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get the importance of each feature in the model.
        
        Returns:
            List[Tuple[str, float]]: List of (feature_name, importance) tuples
            
        Raises:
            Exception: If model doesn't support feature importance
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_imp = list(zip(self.feature_names, importances))
                return sorted(feature_imp, key=lambda x: x[1], reverse=True)
            raise Exception("Model doesn't support feature importance")
        except Exception as e:
            raise Exception(f"Error getting feature importance: {str(e)}")
