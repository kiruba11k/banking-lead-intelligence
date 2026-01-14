"""
Model Predictor - Handles missing data gracefully
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
import json
import os

class ModelPredictor:
    """Handles model predictions with actual data only."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.model_loaded = False
        
        try:
            # Check if model files exist
            model_path = "models/banking_scoring_model_20260113_110158.pkl"
            features_path = "models/banking_scoring_model_20260113_110158_features.pkl"
            metadata_path = "models/banking_scoring_model_20260113_110158_metadata.json"
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return
                
            # Load model
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(self.model)}")
            
            # Load feature names
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                print(f"Feature names loaded: {len(self.feature_names)} features")
            else:
                print(f"Feature names file not found: {features_path}")
                # Try to get feature names from model if available
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = list(self.model.feature_names_in_)
                    print(f"Got feature names from model: {len(self.feature_names)} features")
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("Metadata loaded")
            else:
                print(f"Metadata file not found: {metadata_path}")
                
            self.model_loaded = True
            
        except Exception as e:
            print(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()
    
    def predict(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction with actual data.
        Handles missing values appropriately.
        """
        if not self.model_loaded:
            print("Model not loaded - cannot predict")
            return None
        
        if features_df is None or features_df.empty:
            print("No features provided for prediction")
            return None
        
        try:
            print(f"Features shape: {features_df.shape}")
            print(f"Features columns: {list(features_df.columns)}")
            print(f"Features dtypes: {features_df.dtypes.to_dict()}")
            
            # Check for NaN values
            nan_counts = features_df.isna().sum()
            if nan_counts.any():
                print(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            
            # Handle missing values
            processed_df = self._handle_missing_values(features_df)
            
            # Ensure correct feature order
            if self.feature_names:
                print(f"Expected features: {self.feature_names}")
                
                # Check which features are missing
                missing_features = set(self.feature_names) - set(processed_df.columns)
                if missing_features:
                    print(f"Missing features from input: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        processed_df[feature] = 0
                
                # Check for extra features
                extra_features = set(processed_df.columns) - set(self.feature_names)
                if extra_features:
                    print(f"Extra features in input: {extra_features}")
                    # Remove extra features
                    processed_df = processed_df[self.feature_names]
                else:
                    processed_df = processed_df[self.feature_names]
            
            print(f"Processed features shape: {processed_df.shape}")
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            probabilities = self.model.predict_proba(processed_df)[0]
            
            print(f"Raw prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            
            # Get label mapping from metadata
            label_mapping = self.metadata.get('data_info', {}).get('label_mapping', 
                {"COLD": 0, "COOL": 1, "WARM": 2, "HOT": 3})
            
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            priority = reverse_mapping.get(int(prediction), "UNKNOWN")
            
            print(f"Mapped priority: {priority}")
            
            # Create probability dictionary
            prob_dict = {}
            for label, idx in label_mapping.items():
                if idx < len(probabilities):
                    prob_dict[label] = float(probabilities[idx])
            
            result = {
                "priority": priority,
                "numeric_score": int(prediction),
                "confidence": float(max(probabilities)),
                "probabilities": prob_dict,
                "missing_features": self._get_missing_features(features_df)
            }
            
            print(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        if df is None or df.empty:
            return df
            
        # Create copy
        processed = df.copy()
        
        # Fill missing values with appropriate defaults
        for column in processed.columns:
            if processed[column].isna().any():
                # Check if numeric
                if pd.api.types.is_numeric_dtype(processed[column]):
                    # Use 0 for missing numeric
                    processed[column] = processed[column].fillna(0)
                else:
                    # For categorical, use empty string
                    processed[column] = processed[column].fillna("")
        
        return processed
    
    def _get_missing_features(self, df: pd.DataFrame):
        """Get list of features with missing values."""
        if df is None or df.empty:
            return []
        
        missing = []
        
        for column in df.columns:
            if df[column].isna().any():
                missing.append(column)
        
        return missing
    
    def get_feature_importance(self) -> Optional[Dict]:
        """Get feature importance if available."""
        if not self.model_loaded:
            return None
            
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
