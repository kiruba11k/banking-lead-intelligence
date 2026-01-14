"""
Model Predictor - Handles missing data gracefully
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
import json
from typing import List

class ModelPredictor:
    """Handles model predictions with actual data only."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.metadata = {}
        
        try:
            # Load model
            self.model = joblib.load("models/banking_scoring_model_20260113_110158.pkl")
            
            # Load feature names
            self.feature_names = joblib.load("models/banking_scoring_model_20260113_110158_features.pkl")
            
            # Load metadata
            with open("models/banking_scoring_model_20260113_110158_metadata.json", 'r') as f:
                self.metadata = json.load(f)
                
        except Exception as e:
            print(f"Model loading error: {e}")
    
    def predict(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction with actual data.
        Handles missing values appropriately.
        """
        if self.model is None:
            return None
        
        try:
            # Handle missing values
            processed_df = self._handle_missing_values(features_df)
            
            # Ensure correct feature order
            if self.feature_names:
                processed_df = processed_df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            probabilities = self.model.predict_proba(processed_df)[0]
            
            # Get label mapping from metadata
            label_mapping = self.metadata.get('data_info', {}).get('label_mapping', 
                {"COLD": 0, "COOL": 1, "WARM": 2, "HOT": 3})
            
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            priority = reverse_mapping.get(int(prediction), "UNKNOWN")
            
            # Create probability dictionary
            prob_dict = {}
            for label, idx in label_mapping.items():
                if idx < len(probabilities):
                    prob_dict[label] = float(probabilities[idx])
            
            return {
                "priority": priority,
                "numeric_score": int(prediction),
                "confidence": float(max(probabilities)),
                "probabilities": prob_dict,
                "missing_features": self._get_missing_features(features_df)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Create copy
        processed = df.copy()
        
        # Fill missing values with appropriate defaults
        # For numeric features: median if available, else 0
        # For categorical features: most frequent if available, else empty
        
        for column in processed.columns:
            if processed[column].isna().any():
                # Check if numeric
                if pd.api.types.is_numeric_dtype(processed[column]):
                    # Use 0 for missing numeric (model was trained with this)
                    processed[column] = processed[column].fillna(0)
                else:
                    # For categorical, use empty string
                    processed[column] = processed[column].fillna("")
        
        return processed
    
    def _get_missing_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of features with missing values."""
        missing = []
        
        for column in df.columns:
            if df[column].isna().any():
                missing.append(column)
        
        return missing
    
    def get_feature_importance(self) -> Optional[Dict]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if len(importances) == len(self.feature_names):
                importance_dict = dict(zip(self.feature_names, importances))
                # Sort by importance
                sorted_dict = {k: v for k, v in sorted(
                    importance_dict.items(), key=lambda item: item[1], reverse=True
                )}
                return sorted_dict
        
        return None
