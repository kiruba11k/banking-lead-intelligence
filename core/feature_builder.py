"""
Dynamic Feature Builder - No defaults, only actual data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import re

class DynamicFeatureBuilder:
    """Builds features dynamically from actual data only."""
    
    def __init__(self):
        self.required_features = self._load_required_features()
    
    def _load_required_features(self) -> List[str]:
        """Load required features from model metadata."""
        try:
            import joblib
            features = joblib.load("models/banking_scoring_model_20260113_110158_features.pkl")
            return list(features) if isinstance(features, list) else []
        except:
            # If can't load, return empty list
            return []
    
    def build_features(self, linkedin_data: Optional[Dict], 
                      company_data: Optional[Dict], 
                      user_data: Optional[Dict]) -> Optional[pd.DataFrame]:
        """
        Build features from actual data only.
        Returns None if insufficient data.
        """
        features = {}
        
        # Extract from LinkedIn data
        if linkedin_data:
            self._extract_linkedin_features(features, linkedin_data)
        
        # Extract from company data
        if company_data:
            self._extract_company_features(features, company_data)
        
        # Extract from user input
        if user_data:
            self._extract_user_features(features, user_data)
        
        # Check if we have minimum required data
        if not self._has_minimum_data(features):
            return None
        
        # Create DataFrame with actual features only
        features_df = pd.DataFrame([features])
        
        # Align with model features
        aligned_df = self._align_features(features_df)
        
        return aligned_df
    
    def _extract_linkedin_features(self, features: Dict, data: Dict):
        """Extract features from LinkedIn data."""
        basic_info = data.get('basic_info', {})
        experiences = data.get('experience', [])
        
        # Personal info
        if basic_info.get('fullname'):
            features['full_name'] = basic_info['fullname']
        
        # Current role
        if experiences:
            current_exp = self._get_current_experience(experiences)
            if current_exp:
                if current_exp.get('title'):
                    features['prospect_designation'] = current_exp['title']
                    # Designation features
                    title = current_exp['title'].lower()
                    features['is_ceo'] = 1 if 'ceo' in title or 'chief executive' in title else 0
                    features['is_c_level'] = 1 if any(word in title for word in 
                                                    ['chief', 'cfo', 'cto', 'cio']) else 0
                    features['is_vp'] = 1 if 'vice president' in title or 'vp' in title else 0
                    features['is_director'] = 1 if 'director' in title else 0
                    features['is_manager'] = 1 if 'manager' in title else 0
                
                if current_exp.get('company'):
                    features['current_company'] = current_exp['company']
        
        # Experience metrics
        if experiences:
            total_exp = self._calculate_total_experience(experiences)
            if total_exp > 0:
                features['total_experience_years'] = total_exp
        
        # Location
        if basic_info.get('location', {}).get('full'):
            features['location'] = basic_info['location']['full']
        
        # LinkedIn metrics
        if basic_info.get('connection_count'):
            features['connection_count'] = basic_info['connection_count']
        
        if basic_info.get('follower_count'):
            features['follower_count'] = basic_info['follower_count']
    
    def _get_current_experience(self, experiences: List[Dict]) -> Optional[Dict]:
        """Get current/most recent experience."""
        for exp in experiences:
            if exp.get('is_current', False):
                return exp
        
        return experiences[0] if experiences else None
    
    def _calculate_total_experience(self, experiences: List[Dict]) -> float:
        """Calculate total experience from actual duration strings."""
        total_months = 0
        
        for exp in experiences:
            duration = exp.get('duration', '')
            
            if not duration:
                continue
            
            # Parse duration string like "Apr 2025 - Present · 10 mos"
            # or "Oct 2024 - Nov 2024 · 2 mos"
            if '·' in duration:
                time_part = duration.split('·')[1].strip()
                
                # Extract years
                year_match = re.search(r'(\d+)\s*yrs?', time_part.lower())
                if year_match:
                    total_months += int(year_match.group(1)) * 12
                
                # Extract months
                month_match = re.search(r'(\d+)\s*mos?', time_part.lower())
                if month_match:
                    total_months += int(month_match.group(1))
        
        return round(total_months / 12, 1) if total_months > 0 else 0.0
    
    def _extract_company_features(self, features: Dict, data: Dict):
        """Extract features from company data."""
        # Company size
        if data.get('size'):
            size_str = data['size']
            features['company_size'] = size_str
            
            # Parse size to numeric
            size_numeric = self._parse_size_to_numeric(size_str)
            if size_numeric:
                features['size_numeric'] = size_numeric
                
                # Size categories
                features['size_51_200'] = 1 if 51 <= size_numeric <= 200 else 0
                features['size_201_500'] = 1 if 201 <= size_numeric <= 500 else 0
                features['size_501_1000'] = 1 if 501 <= size_numeric <= 1000 else 0
                features['size_1001_5000'] = 1 if 1001 <= size_numeric <= 5000 else 0
                features['size_5000_plus'] = 1 if size_numeric >= 5000 else 0
        
        # Revenue
        if data.get('revenue'):
            revenue_str = data['revenue']
            revenue_numeric = self._parse_revenue_to_numeric(revenue_str)
            if revenue_numeric:
                features['revenue_millions'] = revenue_numeric
        
        # Industry
        if data.get('industry'):
            industry = data['industry'].lower()
            features['industry'] = data['industry']
            
            # Industry features
            features['is_fintech'] = 1 if 'fintech' in industry else 0
            features['is_commercial_banking'] = 1 if 'commercial' in industry else 0
            features['is_retail_banking'] = 1 if 'retail' in industry else 0
    
    def _parse_size_to_numeric(self, size_str: str) -> Optional[float]:
        """Parse size string to numeric value."""
        if not size_str:
            return None
        
        size_str = str(size_str).lower()
        
        # Handle ranges like "51-200"
        if '-' in size_str:
            parts = size_str.split('-')
            try:
                start = float(parts[0].replace(',', '').replace('k', '000').replace('+', ''))
                end = float(parts[1].replace(',', '').replace('k', '000').replace('+', ''))
                return (start + end) / 2
            except:
                pass
        
        # Handle single numbers
        try:
            num = float(size_str.replace(',', '').replace('k', '000').replace('+', ''))
            return num
        except:
            return None
    
    def _parse_revenue_to_numeric(self, revenue_str: str) -> Optional[float]:
        """Parse revenue string to millions."""
        if not revenue_str:
            return None
        
        revenue_str = str(revenue_str).upper().replace(',', '').replace('$', '').strip()
        
        try:
            if 'B' in revenue_str:
                return float(revenue_str.replace('B', '')) * 1000
            elif 'M' in revenue_str:
                return float(revenue_str.replace('M', ''))
            else:
                val = float(revenue_str)
                return val if val < 1000 else val / 1000
        except:
            return None
    
    def _extract_user_features(self, features: Dict, data: Dict):
        """Extract features from user input."""
        for key, value in data.items():
            if value:  # Only add if value is not empty
                features[key] = value
    
    def _has_minimum_data(self, features: Dict) -> bool:
        """Check if we have minimum required data."""
        # Minimum: designation and some company info
        required_keys = ['prospect_designation']
        
        for key in required_keys:
            if key not in features or not features[key]:
                return False
        
        return True
    
    def _align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Align features with model requirements."""
        if not self.required_features:
            return features_df
        
        # Create empty DataFrame with all required features
        aligned_df = pd.DataFrame(columns=self.required_features)
        
        # Fill with available data
        for feature in self.required_features:
            if feature in features_df.columns:
                aligned_df[feature] = features_df[feature]
            else:
                # Leave as NaN if not available
                aligned_df[feature] = np.nan
        
        return aligned_df
