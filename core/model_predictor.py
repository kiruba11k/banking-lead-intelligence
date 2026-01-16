"""
Model Predictor - Handles features for XGBoost with categorical support
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import json
import os
import warnings

class ModelPredictor:
    """Handles model predictions for banking lead scoring."""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.model_loaded = False
        
        # Expected features from the trained model
        self.expected_features = [
            'is_ceo', 'is_c_level', 'is_evp_svp', 'is_vp', 'is_director', 
            'is_manager', 'is_officer', 'in_lending', 'in_tech', 'in_operations',
            'in_risk', 'in_finance', 'in_strategy', 'designation_length', 
            'size_numeric', 'size_51_200', 'size_201_500', 'size_501_1000', 
            'size_1001_5000', 'size_5000_plus', 'revenue_millions', 
            'revenue_category', 'activity_days', 'is_active_week', 
            'is_active_month', 'is_consumer_lending', 'is_commercial_banking', 
            'is_retail_banking', 'is_fintech', 'is_credit_union'
        ]
        
        # Binary features (should be 0 or 1)
        self.binary_features = [
            'is_ceo', 'is_c_level', 'is_evp_svp', 'is_vp', 'is_director', 
            'is_manager', 'is_officer', 'in_lending', 'in_tech', 'in_operations',
            'in_risk', 'in_finance', 'in_strategy', 'size_51_200', 
            'size_201_500', 'size_501_1000', 'size_1001_5000', 'size_5000_plus',
            'is_active_week', 'is_active_month', 'is_consumer_lending', 
            'is_commercial_banking', 'is_retail_banking', 'is_fintech', 
            'is_credit_union'
        ]
        
        # Numeric features
        self.numeric_features = ['size_numeric', 'revenue_millions', 'activity_days', 'designation_length']
        
        # Categorical features
        self.categorical_features = ['revenue_category']
        
        try:
            # Define paths
            models_dir = "models"
            model_path = os.path.join(models_dir, "banking_scoring_model_20260113_110158.pkl")
            features_path = os.path.join(models_dir, "banking_scoring_model_20260113_110158_features.pkl")
            metadata_path = os.path.join(models_dir, "banking_scoring_model_20260113_110158_metadata.json")
            
            print(f"Looking for model files...")
            
            # Check if files exist
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                self._create_fallback_model()
                return
            
            # Load model
            print(f"Loading model from {model_path}...")
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded. Type: {type(self.model)}")
            
            # Load feature names
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                print(f"✓ Feature names loaded: {len(self.feature_names)} features")
            else:
                print(f"✗ Feature names file not found, using expected features")
                self.feature_names = self.expected_features
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✓ Metadata loaded")
            else:
                print(f"✗ Metadata file not found")
                self.metadata = {
                    'data_info': {
                        'label_mapping': {"COLD": 0, "COOL": 1, "WARM": 2, "HOT": 3}
                    }
                }
            
            self.model_loaded = True
            print(f"✓ Model predictor initialized successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback model."""
        print("Creating fallback model...")
        
        # Use expected features
        self.feature_names = self.expected_features
        
        # Create a simple model that always predicts WARM
        class FallbackModel:
            def predict(self, X):
                return np.array([2] * len(X))  # WARM = 2
            
            def predict_proba(self, X):
                # Return probabilities for 4 classes
                return np.array([[0.1, 0.2, 0.4, 0.3]] * len(X))
        
        self.model = FallbackModel()
        self.metadata = {
            'data_info': {
                'label_mapping': {"COLD": 0, "COOL": 1, "WARM": 2, "HOT": 3}
            }
        }
        self.model_loaded = True
        print("✓ Fallback model created")
    
    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model."""
        if features_df is None or features_df.empty:
            # Create empty DataFrame with expected columns
            return pd.DataFrame(columns=self.feature_names)
        
        # Create a copy
        processed = features_df.copy()
        
        # Ensure all expected features exist
        for feature in self.feature_names:
            if feature not in processed.columns:
                print(f"Adding missing feature: {feature}")
                # Add with appropriate default
                if feature in self.binary_features:
                    processed[feature] = 0
                elif feature in self.numeric_features:
                    processed[feature] = 0.0
                elif feature in self.categorical_features:
                    processed[feature] = '<20M'  # Default category
                else:
                    processed[feature] = 0
        
        # Reorder columns to match model expectations
        processed = processed[self.feature_names]
        
        # Convert binary features to int (0 or 1)
        for feature in self.binary_features:
            if feature in processed.columns:
                try:
                    # Ensure it's 0 or 1
                    processed[feature] = pd.to_numeric(processed[feature], errors='coerce').fillna(0)
                    processed[feature] = processed[feature].apply(lambda x: 1 if x and x != 0 else 0).astype(int)
                except:
                    processed[feature] = 0
        
        # Convert numeric features to float
        for feature in self.numeric_features:
            if feature in processed.columns:
                processed[feature] = pd.to_numeric(processed[feature], errors='coerce').fillna(0)
        
        # Convert categorical features to string then category
        for feature in self.categorical_features:
            if feature in processed.columns:
                processed[feature] = processed[feature].astype(str).astype('category')
        
        # Fill any remaining NaN values
        processed = processed.fillna(0)
        
        print(f"Prepared features shape: {processed.shape}")
        print(f"Feature dtypes: {processed.dtypes.to_dict()}")
        
        return processed
    
    def predict(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Make prediction with actual data.
        """
        if not self.model_loaded:
            print("Model not loaded - cannot predict")
            return self._get_fallback_prediction("Model not loaded")
        
        if features_df is None or features_df.empty:
            print("No features provided for prediction")
            return self._get_fallback_prediction("No features provided")
        
        try:
            print(f"Input features shape: {features_df.shape}")
            print(f"Input columns: {list(features_df.columns)}")
            
            # Prepare features
            processed_df = self._prepare_features(features_df)
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            probabilities = self.model.predict_proba(processed_df)[0]
            
            print(f"Raw prediction: {prediction}")
            print(f"Probabilities: {probabilities}")
            
            # Get label mapping
            label_mapping = self.metadata.get('data_info', {}).get('label_mapping', 
                {"COLD": 0, "COOL": 1, "WARM": 2, "HOT": 3})
            
            reverse_mapping = {v: k for k, v in label_mapping.items()}
            priority = reverse_mapping.get(int(prediction), "UNKNOWN")
            
            # Create probability dictionary
            prob_dict = {}
            for label, idx in label_mapping.items():
                if idx < len(probabilities):
                    prob_dict[label] = float(probabilities[idx])
                else:
                    prob_dict[label] = 0.0
            
            # Calculate confidence
            confidence = float(max(probabilities)) if len(probabilities) > 0 else 0.0
            
            # Get missing features
            missing_features = []
            for feature in self.feature_names:
                if feature not in features_df.columns:
                    missing_features.append(feature)
            
            result = {
                "priority": priority,
                "numeric_score": int(prediction),
                "confidence": confidence,
                "probabilities": prob_dict,
                "missing_features": missing_features,
                "model_type": "xgboost" if not isinstance(self.model, type(self)._create_fallback_model) else "fallback"
            }
            
            print(f"✓ Prediction successful: {result}")
            return result
            
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_prediction(f"Prediction error: {str(e)}")
    
    def _get_fallback_prediction(self, reason: str) -> Dict:
        """Get a fallback prediction."""
        return {
            "priority": "WARM",
            "numeric_score": 2,
            "confidence": 0.5,
            "probabilities": {"COLD": 0.25, "COOL": 0.25, "WARM": 0.25, "HOT": 0.25},
            "missing_features": [],
            "note": reason,
            "model_type": "fallback"
        }
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance from the model.
        Returns a dictionary of feature names and their importance scores.
        """
        if not self.model_loaded:
            print("Model not loaded - cannot get feature importance")
            return None
        
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                if len(importances) == len(self.feature_names):
                    # Create dictionary of feature names and importance scores
                    importance_dict = dict(zip(self.feature_names, importances))
                    
                    # Sort by importance (descending)
                    sorted_dict = {k: v for k, v in sorted(
                        importance_dict.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )}
                    
                    print(f"Feature importance calculated: {len(sorted_dict)} features")
                    return sorted_dict
                else:
                    print(f"Mismatch: Model has {len(importances)} importances, but {len(self.feature_names)} feature names")
                    return None
            else:
                # For XGBoost, try to get importance using get_booster()
                if hasattr(self.model, 'get_booster'):
                    try:
                        import xgboost as xgb
                        booster = self.model.get_booster()
                        importance_dict = booster.get_score(importance_type='weight')
                        
                        # Convert to dictionary with feature names
                        result = {}
                        for feature_idx, importance in importance_dict.items():
                            feature_name = f"f{feature_idx}"
                            # Try to map feature index to actual feature name
                            try:
                                feature_idx_int = int(feature_idx.replace('f', ''))
                                if feature_idx_int < len(self.feature_names):
                                    feature_name = self.feature_names[feature_idx_int]
                            except:
                                pass
                            result[feature_name] = importance
                        
                        # Sort by importance
                        sorted_dict = {k: v for k, v in sorted(
                            result.items(),
                            key=lambda item: item[1],
                            reverse=True
                        )}
                        
                        print(f"XGBoost feature importance calculated")
                        return sorted_dict
                    except Exception as e:
                        print(f"Could not get XGBoost feature importance: {e}")
                        return None
                else:
                    print("Model does not have feature_importances_ attribute")
                    return None
                
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
