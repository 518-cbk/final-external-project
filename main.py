import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="INX Future Inc - Employee Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004080;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        color: #004080;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        color: #0066cc;
        font-size: 2rem;
        margin: 0.5rem 0;
    }
    .metric-card p {
        color: #666666;
        margin-top: 0.5rem;
    }
    .prediction-high {
        background-color: #e6f7e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00b33c;
        box-shadow: 0 2px 6px rgba(0,179,60,0.2);
    }
    .prediction-high h2, .prediction-high h3 {
        color: #006622;
    }
    .prediction-medium {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
        box-shadow: 0 2px 6px rgba(255,153,0,0.2);
    }
    .prediction-medium h2, .prediction-medium h3 {
        color: #cc7a00;
    }
    .prediction-low {
        background-color: #ffe6e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #cc0000;
        box-shadow: 0 2px 6px rgba(204,0,0,0.2);
    }
    .prediction-low h2, .prediction-low h3 {
        color: #990000;
    }
    .nav-button {
        display: block;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        color: #333333;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .nav-button:hover {
        background-color: #e3f2fd;
        border-color: #0066cc;
        transform: translateX(5px);
    }
    .nav-button-active {
        background-color: #0066cc;
        border-color: #004080;
        color: white;
        font-weight: 600;
    }
    .stRadio > label {
        font-size: 1.1rem;
        font-weight: 500;
        color: #333333;
    }
    .stRadio > div {
        gap: 0.5rem;
    }
    .stRadio > div > label {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stRadio > div > label:hover {
        background-color: #e3f2fd;
        border-color: #0066cc;
    }
    .stRadio > div > label[data-checked="true"] {
        background-color: #0066cc;
        border-color: #004080;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class EmployeePerformancePredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.target_encoder = None
        self.scaler = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load the trained model and artifacts"""
        try:
            self.model = joblib.load('best_classification_model_LightGBM.pkl')
            with open('employee_performance_features.json', 'r') as f:
                self.features = json.load(f)['features']
            self.target_encoder = joblib.load('employee_performance_target_encoder.pkl')
            self.scaler = joblib.load('employee_performance_scaler.pkl')
            return True
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess the input data to match training format"""
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # One-hot encode categorical variables (same as training)
        categorical_cols = ['Gender', 'EducationBackground', 'MaritalStatus', 
                          'EmpDepartment', 'BusinessTravelFrequency', 'OverTime', 'Attrition']
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, prefix_sep='_')
        
        # Label encode high cardinality columns (EmpJobRole)
        if 'EmpJobRole' in df.columns:
            le = LabelEncoder()
            df_encoded['EmpJobRole_encoded'] = le.fit_transform(df['EmpJobRole'].astype(str))
            df_encoded = df_encoded.drop(columns=['EmpJobRole'], errors='ignore')
        
        # Ensure all expected features are present
        for feature in self.features:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Select only the features used in training
        X = df_encoded[self.features]
        
        return X
    
    def predict(self, input_data):
        """Make prediction on new employee data"""
        try:
            # Preprocess input
            X_processed = self.preprocess_input(input_data)
            
            # Scale features
            X_scaled = self.scaler.transform(X_processed)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            # Convert back to original label
            predicted_class = self.target_encoder.inverse_transform([prediction])[0]
            
            return predicted_class, probability[prediction], probability
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">📊 INX Future Inc - Employee Performance Intelligence</div>', 
                unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = EmployeePerformancePredictor()
    
    if not predictor.model:
        st.error("❌ Model artifacts not found. Please ensure the model files are in the current directory.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a section",
        ["🏠 Dashboard Overview", "🧪 Performance Predictor", "📈 Business Insights", "🎯 Recommendations"]
    )
    
    if app_mode == "🏠 Dashboard Overview":
        show_dashboard_overview()
    
    elif app_mode == "🧪 Performance Predictor":
        show_performance_predictor(predictor)
    
    elif app_mode == "📈 Business Insights":
        show_business_insights()
    
    elif app_mode == "🎯 Recommendations":
        show_recommendations()

def show_dashboard_overview():
    """Show the main dashboard overview"""
    
    st.markdown('<div class="sub-header">🚀 Executive Summary</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>92.9%</h2>
            <p>Performance Prediction Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Employees Analyzed</h3>
            <h2>1,200</h2>
            <p>Comprehensive Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Key Factors</h3>
            <h2>19</h2>
            <p>Performance Drivers Identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Potential Savings</h3>
            <h2>$667K+</h2>
            <p>Annual Attrition Reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Distribution
    st.markdown('<div class="sub-header">📈 Performance Rating Distribution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create performance distribution chart
        performance_data = {
            'Rating': [2, 3, 4],
            'Count': [194, 874, 132],
            'Percentage': [16.2, 72.8, 11.0]
        }
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.pie(df_perf, values='Count', names='Rating', 
                    title='Employee Performance Distribution',
                    color='Rating',
                    color_discrete_map={'2': '#dc3545', '3': '#ffc107', '4': '#28a745'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Department performance
        dept_data = {
            'Department': ['Development', 'Data Science', 'Human Resources', 
                          'Research & Development', 'Sales', 'Finance'],
            'Performance': [3.09, 3.05, 2.93, 2.92, 2.86, 2.78],
            'Employees': [361, 20, 54, 343, 373, 49]
        }
        df_dept = pd.DataFrame(dept_data)
        
        fig = px.bar(df_dept, x='Performance', y='Department', orientation='h',
                    title='Department Performance Ranking',
                    color='Performance',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performance Drivers
    st.markdown('<div class="sub-header">🎯 Top Performance Drivers</div>', unsafe_allow_html=True)
    
    drivers_data = {
        'Factor': ['Salary Hike %', 'Work Environment', 'Promotion Gap', 
                  'Current Role Experience', 'Job Role', 'Work-Life Balance'],
        'Impact Score': [0.95, 0.86, 0.31, 0.20, 0.18, 0.15],
        'Description': [
            'Recent salary increase percentage',
            'Satisfaction with workplace conditions',
            'Years since last promotion',
            'Experience in current role',
            'Specific job role impact',
            'Work-life balance satisfaction'
        ]
    }
    df_drivers = pd.DataFrame(drivers_data)
    
    fig = px.bar(df_drivers, x='Impact Score', y='Factor', orientation='h',
                title='Top Factors Influencing Employee Performance',
                hover_data=['Description'],
                color='Impact Score',
                color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

def show_performance_predictor(predictor):
    """Show the performance prediction interface"""
    
    st.markdown('<div class="sub-header">🧪 Employee Performance Predictor</div>', unsafe_allow_html=True)
    
    st.info("""
    💡 **How it works**: Enter employee details below to predict their performance rating.
    The AI model analyzes 19 key factors to provide accurate performance predictions.
    """)
    
    # Create input form
    with st.form("employee_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal & Background")
            age = st.slider("Age", 18, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education_background = st.selectbox("Education Background", 
                                              ["Life Sciences", "Medical", "Marketing", 
                                               "Technical Degree", "Human Resources", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        
        with col2:
            st.subheader("Job & Compensation")
            emp_department = st.selectbox("Department", 
                                        ["Sales", "Research & Development", "Human Resources", 
                                         "Finance", "Development", "Data Science"])
            emp_job_role = st.selectbox("Job Role", 
                                      ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                       "Manufacturing Director", "Healthcare Representative", "Manager",
                                       "Research Director", "Human Resources"])
            business_travel = st.selectbox("Business Travel Frequency", 
                                         ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            attrition = st.selectbox("Attrition Risk", ["No", "Yes"])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Satisfaction Metrics")
            environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
            work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        
        with col4:
            st.subheader("Career & Experience")
            distance_from_home = st.slider("Distance from Home (miles)", 1, 30, 10)
            total_experience = st.slider("Total Work Experience (years)", 0, 40, 10)
            years_current_role = st.slider("Years in Current Role", 0, 18, 3)
            years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
            salary_hike_percent = st.slider("Last Salary Hike %", 11, 25, 15)
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Performance")
    
    if submitted:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'EducationBackground': education_background,
            'MaritalStatus': marital_status,
            'EmpDepartment': emp_department,
            'EmpJobRole': emp_job_role,
            'BusinessTravelFrequency': business_travel,
            'DistanceFromHome': distance_from_home,
            'EmpEducationLevel': 3,  # Default value
            'EmpEnvironmentSatisfaction': environment_satisfaction,
            'EmpHourlyRate': 65,  # Default value
            'EmpJobInvolvement': 3,  # Default value
            'EmpJobLevel': 2,  # Default value
            'EmpJobSatisfaction': job_satisfaction,
            'NumCompaniesWorked': 2,  # Default value
            'OverTime': overtime,
            'EmpLastSalaryHikePercent': salary_hike_percent,
            'EmpRelationshipSatisfaction': relationship_satisfaction,
            'TotalWorkExperienceInYears': total_experience,
            'TrainingTimesLastYear': 2,  # Default value
            'EmpWorkLifeBalance': work_life_balance,
            'ExperienceYearsAtThisCompany': 7,  # Default value
            'ExperienceYearsInCurrentRole': years_current_role,
            'YearsSinceLastPromotion': years_since_promotion,
            'YearsWithCurrManager': 4,  # Default value
            'Attrition': attrition
        }
        
        # Make prediction
        with st.spinner("Analyzing employee data..."):
            predicted_class, confidence, probabilities = predictor.predict(input_data)
        
        if predicted_class:
            # Display prediction result
            st.markdown("---")
            st.markdown('<div class="sub-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
            
            # Performance interpretation
            performance_info = {
                2: {"label": "LOW PERFORMANCE", "color": "prediction-low", "icon": "⚠️", "description": "Needs immediate attention and performance improvement plan"},
                3: {"label": "SOLID PERFORMER", "color": "prediction-medium", "icon": "✅", "description": "Reliable performer with potential for growth"},
                4: {"label": "HIGH PERFORMER", "color": "prediction-high", "icon": "🚀", "description": "Exceptional performer - focus on retention and development"}
            }
            
            info = performance_info.get(predicted_class, performance_info[3])
            
            st.markdown(f"""
            <div class="{info['color']}">
                <h2>{info['icon']} Predicted Performance: {predicted_class} - {info['label']}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.subheader("📊 Prediction Confidence Breakdown")
            
            prob_data = []
            for i, prob in enumerate(probabilities):
                class_label = predictor.target_encoder.inverse_transform([i])[0]
                prob_data.append({
                    'Performance Rating': class_label,
                    'Probability': prob,
                    'Label': performance_info[class_label]['label']
                })
            
            df_probs = pd.DataFrame(prob_data)
            fig = px.bar(df_probs, x='Performance Rating', y='Probability',
                        color='Probability', color_continuous_scale='RdYlGn',
                        title='Prediction Probability Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on prediction
            st.subheader("🎯 Recommended Actions")
            
            if predicted_class == 2:  # Low performance
                st.warning("""
                **Performance Improvement Plan Required:**
                - Implement structured coaching and mentoring
                - Set clear performance goals with regular check-ins
                - Provide additional training and resources
                - Consider role realignment if necessary
                """)
            elif predicted_class == 3:  # Solid performer
                st.info("""
                **Development Focus:**
                - Provide growth opportunities and stretch assignments
                - Focus on skill development and career progression
                - Regular feedback and recognition
                - Consider leadership training
                """)
            else:  # High performer
                st.success("""
                **Retention & Growth Strategy:**
                - Implement retention programs and incentives
                - Provide leadership development opportunities
                - Consider accelerated career progression
                - Involve in strategic projects and decision-making
                """)

def show_business_insights():
    """Show business insights and analytics"""
    
    st.markdown('<div class="sub-header">📈 Business Intelligence & Analytics</div>', unsafe_allow_html=True)
    
    # Model Performance
    st.subheader("🤖 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "92.9%", "2.9%")
    with col2:
        st.metric("Precision", "92.8%", "2.8%")
    with col3:
        st.metric("Recall", "92.9%", "2.9%")
    with col4:
        st.metric("F1-Score", "92.8%", "2.8%")
    
    # Attrition Analysis
    st.subheader("📉 Attrition-Performance Correlation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        attrition_data = {
            'Performance Level': ['Low (Rating 2)', 'Solid (Rating 3)', 'High (Rating 4)'],
            'Attrition Rate %': [18.6, 14.2, 13.6]
        }
        df_attrition = pd.DataFrame(attrition_data)
        
        fig = px.bar(df_attrition, x='Performance Level', y='Attrition Rate %',
                    color='Attrition Rate %', color_continuous_scale='Reds',
                    title='Attrition Rates by Performance Level')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cost_data = {
            'Category': ['Current Annual Cost', '25% Reduction Potential', 'Savings Opportunity'],
            'Amount ($)': [2670000, 667000, 2000000]
        }
        df_cost = pd.DataFrame(cost_data)
        
        fig = px.bar(df_cost, x='Category', y='Amount ($)',
                    color='Amount ($)', color_continuous_scale='Greens',
                    title='Attrition Cost Analysis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Department Insights
    st.subheader("🏢 Department Performance Analysis")
    
    dept_analysis = {
        'Department': ['Development', 'Data Science', 'Human Resources', 
                      'Research & Development', 'Sales', 'Finance'],
        'Avg Performance': [3.09, 3.05, 2.93, 2.92, 2.86, 2.78],
        'Employee Count': [361, 20, 54, 343, 373, 49],
        'Status': ['High', 'High', 'Medium', 'Medium', 'Medium', 'Low']
    }
    df_dept = pd.DataFrame(dept_analysis)
    
    fig = px.scatter(df_dept, x='Employee Count', y='Avg Performance',
                    size='Employee Count', color='Status',
                    hover_name='Department', size_max=60,
                    color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
                    title='Department Performance vs Size')
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations():
    """Show strategic recommendations"""
    
    st.markdown('<div class="sub-header">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)
    
    # ROI Analysis
    st.subheader("💰 Initiative ROI Analysis")
    
    initiatives = [
        {"Initiative": "Predictive Hiring Tool", "ROI": "1400%", "Cost": "$60K", "Benefit": "Better hiring decisions"},
        {"Initiative": "Department Training", "ROI": "900%", "Cost": "$100K", "Benefit": "Performance improvement"},
        {"Initiative": "Satisfaction Programs", "ROI": "401%", "Cost": "$80K", "Benefit": "Reduced attrition"}
    ]
    
    for initiative in initiatives:
        with st.expander(f"📊 {initiative['Initiative']} - ROI: {initiative['ROI']}"):
            st.write(f"**Investment:** {initiative['Cost']}")
            st.write(f"**Primary Benefit:** {initiative['Benefit']}")
            st.write("**Expected Impact:** Significant improvement in targeted metrics")
    
    # Implementation Timeline
    st.subheader("📅 Recommended Implementation Timeline")
    
    timeline_data = {
        'Phase': ['Phase 1: Foundation', 'Phase 2: Initiatives', 'Phase 3: Optimization'],
        'Duration': ['Months 1-2', 'Months 3-4', 'Months 5-6'],
        'Key Activities': [
            'Deploy tools, train teams, establish baselines',
            'Department training, satisfaction programs, coaching',
            'Monitor results, retrain model, refine strategy'
        ]
    }
    df_timeline = pd.DataFrame(timeline_data)
    
    st.dataframe(df_timeline, use_container_width=True)
    
    # Key Success Factors
    st.subheader("🔑 Critical Success Factors")
    
    success_factors = [
        "✅ Maintain model transparency for HR compliance",
        "✅ Regular feedback loops with department managers", 
        "✅ Continuous data quality monitoring and validation",
        "✅ Stakeholder engagement in implementation",
        "✅ Regular model retraining with new data",
        "✅ Focus on ethical AI and fair assessments"
    ]
    
    for factor in success_factors:
        st.write(factor)
    
    # Call to Action
    st.markdown("---")
    st.success("""
    **🚀 Ready to Transform Your Employee Performance Management?**
    
    Start by using the Performance Predictor to assess current employees and identify areas for improvement.
    The insights from this dashboard can help you make data-driven decisions to boost productivity,
    reduce attrition, and maintain INX's reputation as a top employer.
    """)

if __name__ == "__main__":
    main()