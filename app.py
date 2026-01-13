"""
Banking Lead Intelligence Platform
Complete automated lead scoring from LinkedIn URLs
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import time
import json
import re
from datetime import datetime
from typing import Dict, Optional, List
import plotly.graph_objects as go

# Import core modules
from core.linkedin_extractor import LinkedInProfileExtractor
from core.company_analyzer import CompanyMetricsAnalyzer
from core.feature_pipeline import LeadFeaturePipeline
from core.model_handler import LeadScoringModel

# Page configuration
st.set_page_config(
    page_title="Banking Lead Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Custom CSS for professional interface
st.markdown("""
    <style>
    .main-header { 
        font-family: 'Helvetica Neue', Arial, sans-serif; 
        color: #1e3a8a; 
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }
    .section-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .data-field {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 6px;
    }
    .confidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
    }
    .high-confidence { background-color: #d1fae5; color: #065f46; }
    .medium-confidence { background-color: #fef3c7; color: #92400e; }
    .low-confidence { background-color: #fee2e2; color: #991b1b; }
    .priority-badge {
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    }
    </style>
    """, unsafe_allow_html=True)

class BankingLeadIntelligenceApp:
    def __init__(self):
        """Initialize the application with all components."""
        self.config = config
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize core components
        self.extractor = None
        self.company_analyzer = CompanyMetricsAnalyzer()
        self.feature_pipeline = LeadFeaturePipeline()
        self.model_handler = LeadScoringModel()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = None
        if 'processed_lead' not in st.session_state:
            st.session_state.processed_lead = None
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1  # 1: Input, 2: Review, 3: Results
        if 'apify_key' not in st.session_state:
            st.session_state.apify_key = ""
    
    def render_header(self):
        """Render the application header."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<h1 class="main-header">Banking Lead Intelligence Platform</h1>', 
                       unsafe_allow_html=True)
            st.markdown("""
                <p style='color: #475569; font-size: 16px; line-height: 1.6;'>
                Automated lead scoring for banking relationship managers. 
                Enter a LinkedIn profile URL to automatically extract professional information, 
                analyze company metrics, and generate predictive lead scores.
                </p>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: right; color: #64748b; font-size: 14px;'>
                <strong>Model Version:</strong> 20260113<br>
                <strong>Accuracy:</strong> 85.2%<br>
                <strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d')}
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
    
    def render_sidebar(self):
        """Render the configuration sidebar."""
        with st.sidebar:
            st.markdown('<h3 style="color: #1e3a8a;">Platform Configuration</h3>', 
                       unsafe_handle_html=True)
            
            # API Configuration
            with st.expander("API Configuration", expanded=True):
                apify_key = st.text_input(
                    "Apify API Key",
                    type="password",
                    value=st.session_state.apify_key,
                    help="Required for LinkedIn profile data extraction",
                    key="apify_key_input"
                )
                
                if apify_key and apify_key != st.session_state.apify_key:
                    st.session_state.apify_key = apify_key
                    self.extractor = LinkedInProfileExtractor(api_key=apify_key)
            
            # Analysis Settings
            with st.expander("Analysis Settings", expanded=True):
                auto_extract = st.checkbox(
                    "Auto-extract all fields",
                    value=True,
                    help="Automatically extract and estimate all required fields"
                )
                
                enable_estimates = st.checkbox(
                    "Enable intelligent estimates",
                    value=True,
                    help="Estimate company metrics when exact data is unavailable"
                )
            
            # System Status
            with st.expander("System Status", expanded=False):
                if self.model_handler.is_loaded():
                    st.success("Model Loaded")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Features", "31")
                    with col2:
                        st.metric("Accuracy", "85.2%")
                else:
                    st.warning("Model Not Loaded")
            
            st.divider()
            
            # Navigation
            st.markdown('<h4 style="color: #1e3a8a;">Navigation</h4>', 
                       unsafe_allow_html=True)
            
            if st.button("New Analysis", use_container_width=True):
                self._reset_analysis()
            
            if st.button("Export Results", use_container_width=True):
                self._export_results()
    
    def render_input_section(self):
        """Render the LinkedIn URL input section."""
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Step 1: LinkedIn Profile Analysis")
        
        # URL Input
        linkedin_url = st.text_input(
            "LinkedIn Profile URL",
            placeholder="https://linkedin.com/in/username",
            help="Enter the complete LinkedIn profile URL",
            key="linkedin_url_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_clicked = st.button(
                "Analyze Profile",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.apify_key
            )
        
        with col2:
            if st.button("Use Sample Profile", use_container_width=True):
                linkedin_url = "https://linkedin.com/in/kirubakaranperiyasamy"
                analyze_clicked = True
        
        if not st.session_state.apify_key:
            st.warning("Please enter your Apify API key in the sidebar to begin analysis.")
        
        if analyze_clicked and linkedin_url:
            if not st.session_state.apify_key:
                st.error("Apify API key is required for profile analysis.")
                return
            
            self._analyze_profile(linkedin_url)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _analyze_profile(self, linkedin_url: str):
        """Analyze LinkedIn profile and extract data."""
        
        # Initialize extractor
        if not self.extractor:
            self.extractor = LinkedInProfileExtractor(api_key=st.session_state.apify_key)
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Step 1: Extract LinkedIn data
            st.info("Step 1/3: Extracting LinkedIn profile data...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Extract data from LinkedIn
                status_text.text("Connecting to Apify API...")
                progress_bar.progress(20)
                
                extracted_data = self.extractor.extract_profile_data(linkedin_url)
                
                if not extracted_data:
                    st.error("Failed to extract LinkedIn profile data.")
                    return
                
                # Step 2: Analyze company metrics
                status_text.text("Step 2/3: Analyzing company information...")
                progress_bar.progress(50)
                
                # Get current role (most recent experience)
                current_role = self._extract_current_role(extracted_data)
                
                if not current_role:
                    st.error("No professional experience found in profile.")
                    return
                
                # Analyze company from current role
                company_analysis = self.company_analyzer.analyze_company(
                    current_role.get('company', ''),
                    extracted_data
                )
                
                # Step 3: Process lead data
                status_text.text("Step 3/3: Processing lead information...")
                progress_bar.progress(80)
                
                processed_lead = self._process_lead_data(
                    extracted_data, 
                    current_role, 
                    company_analysis
                )
                
                # Store in session state
                st.session_state.extracted_data = extracted_data
                st.session_state.processed_lead = processed_lead
                st.session_state.current_step = 2
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Rerun to show next step
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Analysis failed: {str(e)}")
    
    def _extract_current_role(self, extracted_data: Dict) -> Optional[Dict]:
        """Extract the current/most recent role from experience data."""
        experiences = extracted_data.get('experience', [])
        
        if not experiences:
            return None
        
        # First, look for current positions
        current_positions = [
            exp for exp in experiences 
            if exp.get('is_current', False) == True
        ]
        
        if current_positions:
            # Return most recent current position
            return current_positions[0]
        
        # If no current position, return most recent past position
        # Sort by start date (assuming most recent is first)
        return experiences[0] if experiences else None
    
    def _process_lead_data(self, extracted_data: Dict, current_role: Dict, 
                          company_analysis: Dict) -> Dict:
        """Process all extracted data into lead format."""
        
        basic_info = extracted_data.get('basic_info', {})
        
        # Build lead data structure
        lead_data = {
            # Basic Information
            'linkedin_url': basic_info.get('profile_url', ''),
            'extraction_timestamp': datetime.now().isoformat(),
            
            # Personal Information
            'full_name': basic_info.get('fullname', ''),
            'first_name': basic_info.get('first_name', ''),
            'last_name': basic_info.get('last_name', ''),
            'headline': basic_info.get('headline', ''),
            'about': basic_info.get('about', ''),
            'location': basic_info.get('location', {}).get('full', ''),
            
            # Current Role Information
            'prospect_designation': current_role.get('title', ''),
            'current_company': current_role.get('company', ''),
            'role_location': current_role.get('location', ''),
            'role_description': current_role.get('description', ''),
            'role_start_date': current_role.get('start_date', {}),
            'is_current_role': current_role.get('is_current', False),
            
            # Company Analysis
            'company_size': company_analysis.get('estimated_size', '51-200'),
            'annual_revenue': company_analysis.get('estimated_revenue', '$10M-$50M'),
            'industry_sector': company_analysis.get('industry', 'Technology'),
            'company_reputation_score': company_analysis.get('reputation_score', 5),
            
            # Career Metrics
            'total_experience_years': self._calculate_experience_years(
                extracted_data.get('experience', [])
            ),
            'current_tenure_months': self._calculate_current_tenure(current_role),
            'education_level': self._get_education_level(
                extracted_data.get('education', [])
            ),
            
            # LinkedIn Metrics
            'connection_count': basic_info.get('connection_count', 0),
            'follower_count': basic_info.get('follower_count', 0),
            'is_premium_member': basic_info.get('is_premium', False),
            
            # Skills & Certifications
            'top_skills': self._extract_top_skills(extracted_data),
            'certification_count': len(extracted_data.get('certifications', [])),
            'project_count': len(extracted_data.get('projects', [])),
            
            # Confidence Scores
            'extraction_confidence': 'high',
            'company_estimation_confidence': company_analysis.get('confidence', 'medium'),
            
            # Raw data reference
            'raw_experience': extracted_data.get('experience', []),
            'raw_education': extracted_data.get('education', [])
        }
        
        return lead_data
    
    def _calculate_experience_years(self, experiences: List[Dict]) -> float:
        """Calculate total years of professional experience."""
        if not experiences:
            return 0.0
        
        total_months = 0
        
        for exp in experiences:
            duration = exp.get('duration', '')
            
            # Parse duration strings
            if 'mos' in duration.lower():
                # Extract months
                match = re.search(r'(\d+)\s*mos', duration)
                if match:
                    total_months += int(match.group(1))
            elif 'yrs' in duration.lower():
                # Extract years and convert to months
                match = re.search(r'(\d+)\s*yrs', duration)
                if match:
                    total_months += int(match.group(1)) * 12
                # Check for additional months
                month_match = re.search(r'(\d+)\s*mo', duration)
                if month_match:
                    total_months += int(month_match.group(1))
        
        # Convert to years
        return round(total_months / 12, 1)
    
    def _calculate_current_tenure(self, current_role: Dict) -> int:
        """Calculate tenure in current role in months."""
        start_date = current_role.get('start_date', {})
        
        if not start_date:
            return 0
        
        start_year = start_date.get('year', datetime.now().year)
        start_month = self._month_to_number(start_date.get('month', 'Jan'))
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        return (current_year - start_year) * 12 + (current_month - start_month)
    
    def _month_to_number(self, month: str) -> int:
        """Convert month name to number."""
        months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        return months.get(month[:3], 1)
    
    def _get_education_level(self, education: List[Dict]) -> str:
        """Determine highest education level."""
        if not education:
            return 'Bachelor\'s'
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            if 'phd' in degree or 'doctor' in degree:
                return 'PhD'
            elif 'master' in degree or 'mba' in degree or 'ms' in degree:
                return 'Master\'s'
            elif 'bachelor' in degree or 'be' in degree or 'bs' in degree:
                return 'Bachelor\'s'
        
        return 'Bachelor\'s'
    
    def _extract_top_skills(self, extracted_data: Dict) -> List[str]:
        """Extract top skills from profile."""
        skills = []
        
        # Extract from experiences
        for exp in extracted_data.get('experience', []):
            exp_skills = exp.get('skills', [])
            if isinstance(exp_skills, list):
                skills.extend([s for s in exp_skills if s not in skills])
        
        # Limit to top 10
        return skills[:10]
    
    def render_review_section(self):
        """Render the data review and adjustment section."""
        if not st.session_state.processed_lead:
            return
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Step 2: Review & Adjust Extracted Data")
        
        lead_data = st.session_state.processed_lead
        
        # Personal Information
        with st.expander("Personal Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Full Name", value=lead_data['full_name'], disabled=True)
                st.text_input("Headline", value=lead_data['headline'], disabled=True)
            
            with col2:
                st.text_input("Location", value=lead_data['location'], disabled=True)
                st.text_input("LinkedIn URL", value=lead_data['linkedin_url'], disabled=True)
        
        # Professional Information
        with st.expander("Professional Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                designation = st.text_input(
                    "Current Designation",
                    value=lead_data['prospect_designation'],
                    key="designation_input"
                )
                company = st.text_input(
                    "Current Company",
                    value=lead_data['current_company'],
                    key="company_input"
                )
            
            with col2:
                total_exp = st.number_input(
                    "Total Experience (years)",
                    min_value=0.0,
                    max_value=50.0,
                    value=lead_data['total_experience_years'],
                    step=0.5,
                    key="exp_input"
                )
                tenure = st.number_input(
                    "Current Tenure (months)",
                    min_value=0,
                    max_value=600,
                    value=lead_data['current_tenure_months'],
                    key="tenure_input"
                )
        
        # Company Information
        with st.expander("Company Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Company size selection
                size_options = ["1-10", "11-50", "51-200", "201-500", 
                              "501-1000", "1001-5000", "5000+"]
                current_size = lead_data['company_size']
                size_index = size_options.index(current_size) if current_size in size_options else 2
                
                company_size = st.selectbox(
                    "Company Size",
                    options=size_options,
                    index=size_index,
                    key="size_input"
                )
                
                # Update revenue based on size
                revenue_map = config['estimation']['employee_ranges']
                estimated_revenue = revenue_map.get(company_size, "$10M-$50M")
            
            with col2:
                # Industry selection
                industry_options = [
                    "FinTech", "Commercial Banking", "Retail Banking",
                    "Investment Banking", "Insurance", "Asset Management",
                    "Technology", "Consulting", "Healthcare", "Other Financial"
                ]
                current_industry = lead_data['industry_sector']
                industry_index = industry_options.index(current_industry) if current_industry in industry_options else 6
                
                industry = st.selectbox(
                    "Industry Sector",
                    options=industry_options,
                    index=industry_index,
                    key="industry_input"
                )
                
                # Revenue display (read-only based on size)
                st.text_input(
                    "Estimated Annual Revenue",
                    value=estimated_revenue,
                    disabled=True,
                    help="Estimated based on company size"
                )
        
        # Generate Score Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Lead Score", type="primary", use_container_width=True):
                # Update lead data with adjustments
                updated_lead = lead_data.copy()
                updated_lead.update({
                    'prospect_designation': designation,
                    'current_company': company,
                    'total_experience_years': total_exp,
                    'current_tenure_months': tenure,
                    'company_size': company_size,
                    'annual_revenue': estimated_revenue,
                    'industry_sector': industry,
                    'user_adjusted': True,
                    'adjustment_timestamp': datetime.now().isoformat()
                })
                
                st.session_state.processed_lead = updated_lead
                self._generate_prediction(updated_lead)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _generate_prediction(self, lead_data: Dict):
        """Generate lead score prediction."""
        with st.spinner("Generating predictive score..."):
            try:
                # Process through feature pipeline
                features = self.feature_pipeline.transform(lead_data)
                
                # Generate prediction
                prediction_result = self.model_handler.predict(features)
                
                # Store result
                st.session_state.prediction_result = prediction_result
                st.session_state.current_step = 3
                
                # Rerun to show results
                st.rerun()
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    def render_results_section(self):
        """Render the prediction results section."""
        if not st.session_state.prediction_result:
            return
        
        prediction = st.session_state.prediction_result
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Step 3: Lead Scoring Results")
        
        # Priority Display
        priority = prediction['priority']
        confidence = prediction['confidence']
        color = config['display']['priority_colors'].get(priority, "#64748b")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
                <div style='border-left: 5px solid {color}; padding-left: 20px;'>
                    <h2 style='color: {color}; margin: 0;'>Priority: {priority}</h2>
                    <p style='color: #475569; font-size: 16px;'>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            score_value = prediction['probabilities'].get(priority, 0)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='font-size: 12px; color: #64748b;'>SCORE</div>
                    <div style='font-size: 42px; font-weight: bold; color: {color};'>
                        {score_value:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(float(confidence))
        
        # Probability Distribution
        st.markdown("#### Probability Distribution")
        
        prob_df = pd.DataFrame({
            "Priority": ["COLD", "COOL", "WARM", "HOT"],
            "Probability": [
                prediction['probabilities'].get("COLD", 0),
                prediction['probabilities'].get("COOL", 0),
                prediction['probabilities'].get("WARM", 0),
                prediction['probabilities'].get("HOT", 0)
            ]
        })
        
        # Create chart
        fig = go.Figure(data=[
            go.Bar(
                x=prob_df["Priority"],
                y=prob_df["Probability"],
                marker_color=[
                    config['display']['priority_colors']["COLD"],
                    config['display']['priority_colors']["COOL"],
                    config['display']['priority_colors']["WARM"],
                    config['display']['priority_colors']["HOT"]
                ],
                text=[f"{p:.1%}" for p in prob_df["Probability"]],
                textposition="auto",
            )
        ])
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Factors
        if hasattr(self.model_handler.model, 'feature_importances_'):
            st.markdown("#### Key Influencing Factors")
            
            feature_importance = self.model_handler.get_feature_importance()
            if feature_importance is not None:
                top_features = feature_importance.head(5)
                
                for feature, importance in top_features.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(feature)
                    with col2:
                        st.progress(float(importance))
        
        # Action Recommendations
        self._render_action_recommendations(priority, confidence)
        
        # Export Options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Results", use_container_width=True):
                self._export_results()
        
        with col2:
            if st.button("New Analysis", use_container_width=True):
                self._reset_analysis()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_action_recommendations(self, priority: str, confidence: float):
        """Render action recommendations based on priority."""
        
        recommendations = {
            "HOT": {
                "title": "Immediate Action Required",
                "timeline": "Within 24 hours",
                "actions": [
                    "Contact via phone and personalized email",
                    "Prepare tailored commercial banking proposal",
                    "Schedule executive introduction meeting",
                    "Initiate credit pre-approval process"
                ]
            },
            "WARM": {
                "title": "Strategic Nurturing",
                "timeline": "Within 1 week",
                "actions": [
                    "Add to targeted email sequence",
                    "Schedule introductory call within 3-5 days",
                    "Share relevant industry insights",
                    "Connect on LinkedIn with personalized message"
                ]
            },
            "COOL": {
                "title": "Long-term Cultivation",
                "timeline": "Quarterly engagement",
                "actions": [
                    "Add to newsletter distribution",
                    "Monitor for business changes or funding rounds",
                    "Send quarterly market updates",
                    "Re-evaluate in 90 days"
                ]
            },
            "COLD": {
                "title": "Database Maintenance",
                "timeline": "Annual review",
                "actions": [
                    "Include in annual market communications",
                    "Verify contact information yearly",
                    "Monitor for significant changes",
                    "Consider for broad campaigns"
                ]
            }
        }
        
        guide = recommendations.get(priority, recommendations["COOL"])
        
        st.markdown("#### Recommended Action Plan")
        
        with st.container():
            st.markdown(f"""
                <div style='background-color: #f8fafc; padding: 20px; border-radius: 8px; border-left: 4px solid {config['display']['priority_colors'].get(priority, "#64748b")};'>
                    <h4 style='color: #1e293b; margin-top: 0;'>{guide['title']}</h4>
                    <div style='color: #475569; margin-bottom: 15px;'>
                        <strong>Timeline:</strong> {guide['timeline']}
                    </div>
                    <div style='color: #334155;'>
                        <strong>Key Actions:</strong>
                        <ul style='margin-top: 10px;'>
                            {''.join([f'<li>{action}</li>' for action in guide['actions']])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _reset_analysis(self):
        """Reset the analysis session."""
        st.session_state.extracted_data = None
        st.session_state.processed_lead = None
        st.session_state.prediction_result = None
        st.session_state.current_step = 1
        st.rerun()
    
    def _export_results(self):
        """Export analysis results."""
        if not st.session_state.prediction_result:
            st.warning("No results to export.")
            return
        
        export_data = {
            "lead_data": st.session_state.processed_lead,
            "prediction": st.session_state.prediction_result,
            "extraction_data": st.session_state.extracted_data,
            "export_timestamp": datetime.now().isoformat(),
            "model_version": "20260113"
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="Download Complete Analysis (JSON)",
            data=json_str,
            file_name=f"lead_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def run(self):
        """Main application execution."""
        self.render_header()
        
        # Render based on current step
        if st.session_state.current_step == 1:
            self.render_input_section()
        elif st.session_state.current_step == 2:
            self.render_review_section()
        elif st.session_state.current_step == 3:
            self.render_results_section()
        
        # Always render sidebar
        self.render_sidebar()

# Run the application
if __name__ == "__main__":
    app = BankingLeadIntelligenceApp()
    app.run()
