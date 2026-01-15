"""
Dynamic Feature Builder - Creates features matching the trained model
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import re
from datetime import datetime

class DynamicFeatureBuilder:
    """Builds features from LinkedIn and company data."""
    
    def build_features(self, linkedin_data: Optional[Dict] = None, 
                       company_data: Optional[Dict] = None,
                       user_data: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Build features from all available data sources.
        Returns a DataFrame with exactly the features the model expects.
        """
        try:
            print("Building features from available data...")
            
            # Start with empty features dictionary
            features = {}
            
            # 1. Designation-based features (from LinkedIn)
            if linkedin_data:
                features.update(self._extract_designation_features(linkedin_data))
            
            # 2. Company size features
            if company_data or user_data:
                features.update(self._extract_size_features(company_data, user_data))
            
            # 3. Revenue features
            if company_data or user_data:
                features.update(self._extract_revenue_features(company_data, user_data))
            
            # 4. Activity features (from LinkedIn)
            if linkedin_data:
                features.update(self._extract_activity_features(linkedin_data))
            
            # 5. Industry features
            if company_data or user_data:
                features.update(self._extract_industry_features(company_data, user_data))
            
            # Create DataFrame with ALL expected features (set to 0 if not available)
            all_expected_features = [
                'is_ceo', 'is_c_level', 'is_evp_svp', 'is_vp', 'is_director', 
                'is_manager', 'is_officer', 'in_lending', 'in_tech', 'in_operations',
                'in_risk', 'in_finance', 'in_strategy', 'designation_length', 
                'size_numeric', 'size_51_200', 'size_201_500', 'size_501_1000', 
                'size_1001_5000', 'size_5000_plus', 'revenue_millions', 
                'revenue_category', 'activity_days', 'is_active_week', 
                'is_active_month', 'is_consumer_lending', 'is_commercial_banking', 
                'is_retail_banking', 'is_fintech', 'is_credit_union'
            ]
            
            # Ensure all features exist (fill missing with 0)
            for feature in all_expected_features:
                if feature not in features:
                    features[feature] = 0
            
            # Create DataFrame
            features_df = pd.DataFrame([features])
            
            # Convert ALL features to integers (0 or 1) for binary features
            binary_features = [
                'is_ceo', 'is_c_level', 'is_evp_svp', 'is_vp', 'is_director', 
                'is_manager', 'is_officer', 'in_lending', 'in_tech', 'in_operations',
                'in_risk', 'in_finance', 'in_strategy', 'size_51_200', 
                'size_201_500', 'size_501_1000', 'size_1001_5000', 'size_5000_plus',
                'is_active_week', 'is_active_month', 'is_consumer_lending', 
                'is_commercial_banking', 'is_retail_banking', 'is_fintech', 
                'is_credit_union'
            ]
            
            for feature in binary_features:
                if feature in features_df.columns:
                    features_df[feature] = features_df[feature].astype(int)
            
            # Convert numeric features to float
            numeric_features = ['size_numeric', 'revenue_millions', 'activity_days', 'designation_length']
            for feature in numeric_features:
                if feature in features_df.columns:
                    features_df[feature] = pd.to_numeric(features_df[feature], errors='coerce').fillna(0)
            
            print(f"Built {features_df.shape[1]} features for model")
            print(f"Feature dtypes: {features_df.dtypes.to_dict()}")
            
            return features_df
            
        except Exception as e:
            print(f"Error building features: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_designation_features(self, linkedin_data: Dict) -> Dict:
        """Extract designation features from LinkedIn data."""
        features = {}
        
        # Get current position from LinkedIn
        current_position = self._get_current_position(linkedin_data)
        designation = current_position.get('title', '').lower() if current_position else ''
        
        # Seniority level features (must be 0 or 1)
        features['is_ceo'] = 1 if any(term in designation for term in ['ceo', 'chief executive', 'president']) else 0
        features['is_c_level'] = 1 if any(term in designation for term in ['chief', 'cto', 'cfo', 'cio', 'cro', 'cmo']) else 0
        features['is_evp_svp'] = 1 if any(term in designation for term in ['evp', 'svp', 'executive vice president', 'senior vice president']) else 0
        features['is_vp'] = 1 if any(term in designation for term in ['vice president', 'vp', 'v.p.']) else 0
        features['is_director'] = 1 if any(term in designation for term in ['director', 'head of']) else 0
        features['is_manager'] = 1 if any(term in designation for term in ['manager', 'lead', 'supervisor']) else 0
        features['is_officer'] = 1 if any(term in designation for term in ['officer', 'avp', 'assistant vice president']) else 0
        
        # Department/function features
        features['in_lending'] = 1 if any(term in designation for term in ['lend', 'mortgage', 'loan', 'credit']) else 0
        features['in_tech'] = 1 if any(term in designation for term in ['tech', 'technology', 'it', 'digital', 'data', 'analytics', 'ai', 'software']) else 0
        features['in_operations'] = 1 if any(term in designation for term in ['operat', 'process', 'delivery', 'service', 'support']) else 0
        features['in_risk'] = 1 if any(term in designation for term in ['risk', 'compliance', 'security', 'audit']) else 0
        features['in_finance'] = 1 if any(term in designation for term in ['finance', 'fpa', 'treasury', 'cfo']) else 0
        features['in_strategy'] = 1 if any(term in designation for term in ['strategy', 'transformation', 'innovation', 'growth']) else 0
        
        # Title length features
        features['designation_length'] = len(designation)
        
        return features
    
    def _get_current_position(self, linkedin_data: Dict) -> Optional[Dict]:
        """Get current position from LinkedIn data."""
        experiences = linkedin_data.get('experience', [])
        
        if not experiences:
            return None
        
        # Find current position
        for exp in experiences:
            if exp.get('is_current', False):
                return exp
        
        # If no current, use most recent
        return experiences[0] if experiences else None
    
    def _extract_size_features(self, company_data: Optional[Dict], user_data: Optional[Dict]) -> Dict:
        """Extract company size features."""
        features = {}
        
        # Get size from company data or user input
        size_str = ""
        if company_data and 'size' in company_data:
            size_str = str(company_data['size']).lower()
        elif user_data and 'company_size' in user_data:
            size_str = str(user_data['company_size']).lower()
        
        # Parse size to numeric
        features['size_numeric'] = self._parse_size_to_number(size_str)
        
        # Size categories (binary features)
        size_numeric = features['size_numeric']
        features['size_51_200'] = 1 if 51 <= size_numeric <= 200 else 0
        features['size_201_500'] = 1 if 201 <= size_numeric <= 500 else 0
        features['size_501_1000'] = 1 if 501 <= size_numeric <= 1000 else 0
        features['size_1001_5000'] = 1 if 1001 <= size_numeric <= 5000 else 0
        features['size_5000_plus'] = 1 if size_numeric >= 5000 else 0
        
        return features
    
    def _parse_size_to_number(self, size_str: str) -> int:
        """Convert size string to approximate employee count."""
        if not size_str:
            return 1000  # Default
        
        size_str = str(size_str).lower()
        
        # Handle ranges like "51-200"
        if '-' in size_str:
            parts = size_str.split('-')
            try:
                # Take the midpoint of the range
                start = float(parts[0].replace(',', '').replace('k', '000').replace('employees', ''))
                end = float(parts[1].replace(',', '').replace('k', '000').replace('employees', ''))
                return int((start + end) / 2)
            except:
                return 1000
        
        # Handle "500+", "1000+" etc.
        if '+' in size_str:
            try:
                return int(float(size_str.replace('+', '').replace(',', '').replace('k', '000')))
            except:
                return 1000
        
        # Handle standalone numbers
        try:
            num = float(size_str.replace(',', '').replace('k', '000'))
            return int(num)
        except:
            return 1000
    
    def _extract_revenue_features(self, company_data: Optional[Dict], user_data: Optional[Dict]) -> Dict:
        """Extract revenue features."""
        features = {}
        
        # Get revenue from company data or user input
        revenue_str = ""
        if company_data and 'revenue' in company_data:
            revenue_str = str(company_data['revenue'])
        elif user_data and 'annual_revenue' in user_data:
            revenue_str = str(user_data['annual_revenue'])
        
        # Parse revenue to numeric (in millions)
        features['revenue_millions'] = self._parse_revenue_to_millions(revenue_str)
        
        # Revenue category (string, will be converted to category in model predictor)
        revenue = features['revenue_millions']
        if revenue < 20:
            features['revenue_category'] = '<20M'
        elif revenue < 50:
            features['revenue_category'] = '20-50M'
        elif revenue < 100:
            features['revenue_category'] = '50-100M'
        elif revenue < 500:
            features['revenue_category'] = '100-500M'
        else:
            features['revenue_category'] = '500M+'
        
        return features
    
    def _parse_revenue_to_millions(self, revenue_str: str) -> float:
        """Convert revenue string to numeric (in millions)."""
        if not revenue_str:
            return 0.0
        
        s = str(revenue_str).upper().replace(',', '').replace('$', '').strip()
        
        # Regex to find a number and an optional unit
        match = re.match(r'(\d+(\.\d+)?)\s*(B|M|BILLION|MILLION|ILLION)?', s)
        
        if match:
            numeric_val = float(match.group(1))
            unit_str = match.group(3)
            
            if unit_str == 'B' or unit_str == 'BILLION':
                return numeric_val * 1000  # Billions to millions
            elif unit_str in ['M', 'MILLION', 'ILLION']:
                return numeric_val
            else:
                # No explicit unit
                return numeric_val if numeric_val < 1000 or '.' in str(numeric_val) else numeric_val / 1000
        else:
            try:
                val = float(s)
                return val if val < 1000 or '.' in str(val) else val / 1000
            except ValueError:
                return 0.0
    
    def _extract_activity_features(self, linkedin_data: Dict) -> Dict:
        """Extract activity features from LinkedIn."""
        features = {}
        
        # For now, use default values since LinkedIn API might not provide activity
        # You can enhance this with actual LinkedIn activity data if available
        features['activity_days'] = 30  # Default: last active 30 days ago
        features['is_active_week'] = 0  # Default: not active in last week
        features['is_active_month'] = 1  # Default: active in last month
        
        return features
    
    def _extract_industry_features(self, company_data: Optional[Dict], user_data: Optional[Dict]) -> Dict:
        """Extract industry features."""
        features = {}
        
        # Get industry from company data or user input
        industry_str = ""
        if company_data and 'industry' in company_data:
            industry_str = str(company_data['industry']).lower()
        elif user_data and 'industry' in user_data:
            industry_str = str(user_data['industry']).lower()
        
        # Industry sub-sectors (binary features)
        features['is_consumer_lending'] = 1 if any(term in industry_str for term in ['consumer lending', 'consumer finance', 'mortgage lending']) else 0
        features['is_commercial_banking'] = 1 if any(term in industry_str for term in ['commercial', 'business banking', 'corporate banking']) else 0
        features['is_retail_banking'] = 1 if any(term in industry_str for term in ['retail', 'personal banking']) else 0
        features['is_fintech'] = 1 if any(term in industry_str for term in ['fintech', 'financial technology', 'digital banking']) else 0
        features['is_credit_union'] = 1 if any(term in industry_str for term in ['credit union', 'cu', 'cooperative']) else 0
        
        return features
