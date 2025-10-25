import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Traffic Accident Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model_and_metadata():
    try:
        model = joblib.load('models/model.pkl')
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, metadata = load_model_and_metadata()

# Title and header
st.title("🚗 Traffic Accident Risk Predictor")
st.markdown("### AI-Powered Risk Assessment with Advanced Analytics")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    
    if metadata:
        st.metric("Model", metadata['best_model'])
        st.metric("Accuracy", f"{metadata['best_model_metrics']['accuracy']*100:.2f}%")
        st.metric("F1-Score", f"{metadata['best_model_metrics']['f1_score']*100:.2f}%")
        st.metric("Precision", f"{metadata['best_model_metrics']['precision']*100:.2f}%")
        st.metric("Recall", f"{metadata['best_model_metrics']['recall']*100:.2f}%")
        
        st.divider()
        st.caption(f"Training Date: {metadata['training_date']}")
        st.caption(f"Train Size: {metadata['dataset']['train_size']}")
        st.caption(f"Test Size: {metadata['dataset']['test_size']}")

# Main content
if model and metadata:
    FEATURES = metadata['dataset']['features']
    CATEGORICAL = metadata['dataset']['categorical_features']
    NUMERICAL = metadata['dataset']['numerical_features']
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Enter Accident Scenario Details")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        with col1:
            st.subheader("⏰ Time Information")
            
            if 'hour' in FEATURES:
                input_data['hour'] = st.slider("Hour of Day", 0, 23, 12)
            if 'day_of_week' in FEATURES:
                input_data['day_of_week'] = st.slider("Day of Week (0=Mon)", 0, 6, 3)
            if 'month' in FEATURES:
                input_data['month'] = st.slider("Month", 1, 12, 6)
            if 'is_weekend' in FEATURES:
                input_data['is_weekend'] = st.selectbox("Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'is_rush_hour' in FEATURES:
                input_data['is_rush_hour'] = st.selectbox("Rush Hour?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'is_night' in FEATURES:
                input_data['is_night'] = st.selectbox("Night Time?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'is_monsoon' in FEATURES:
                input_data['is_monsoon'] = st.selectbox("Monsoon Season?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        with col2:
            st.subheader("🌦️ Weather & Environment")
            
            if 'weather_main' in FEATURES:
                input_data['weather_main'] = st.selectbox(
                    "Weather Condition",
                    ['Clear', 'Rain', 'Fog', 'Cloudy', 'Snow', 'Storm', 'Other']
                )
            if 'road_type' in FEATURES:
                input_data['road_type'] = st.selectbox(
                    "Road Type",
                    ['highway', 'arterial', 'urban', 'rural', 'residential', 'unknown']
                )
        
        with col3:
            st.subheader("🛣️ Road & Accident Details")
            
            if 'is_junction' in FEATURES:
                input_data['is_junction'] = st.selectbox("At Junction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'is_urban' in FEATURES:
                input_data['is_urban'] = st.selectbox("Urban Area?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'is_highway' in FEATURES:
                input_data['is_highway'] = st.selectbox("Highway?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            if 'Year' in FEATURES:
                input_data['Year'] = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
            if 'year' in FEATURES:
                input_data['year'] = st.number_input("year (lowercase)", min_value=2000, max_value=2030, value=2023)
            if 'Number of Vehicles Involved' in FEATURES:
                input_data['Number of Vehicles Involved'] = st.number_input("Number of Vehicles", min_value=0, max_value=10, value=2)
            if 'Number of Casualties' in FEATURES:
                input_data['Number of Casualties'] = st.number_input("Number of Casualties", min_value=0, max_value=20, value=0)
            if 'Number of Fatalities' in FEATURES:
                input_data['Number of Fatalities'] = st.number_input("Number of Fatalities", min_value=0, max_value=10, value=0)
            if 'Speed Limit (km/h)' in FEATURES:
                input_data['Speed Limit (km/h)'] = st.number_input("Speed Limit (km/h)", min_value=0, max_value=150, value=60)
            if 'Driver Age' in FEATURES:
                input_data['Driver Age'] = st.number_input("Driver Age", min_value=16, max_value=100, value=35)
        
        # Prediction button
        st.divider()
        
        if st.button("🔮 Predict Risk Level", type="primary", use_container_width=True):
            # Create dataframe with proper order
            df_input = pd.DataFrame([{feat: input_data.get(feat, 0) for feat in FEATURES}])
            
            # Make prediction
            try:
                prediction_proba = model.predict_proba(df_input)[0]
                prediction_class = int(model.predict(df_input)[0])
                risk_probability = float(prediction_proba[1]) * 100
                
                # Determine risk level
                if risk_probability >= 70:
                    risk_level = "🔴 High Risk"
                    risk_color = "#f5576c"
                    risk_desc = "⚠️ High accident risk detected! Exercise extreme caution."
                elif risk_probability >= 40:
                    risk_level = "🟡 Medium Risk"
                    risk_color = "#fee140"
                    risk_desc = "⚡ Moderate risk - Exercise caution and stay alert."
                else:
                    risk_level = "🟢 Low Risk"
                    risk_color = "#48bb78"
                    risk_desc = "✅ Low risk - Conditions appear relatively safe."
                
                # Display results
                st.success(risk_desc)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Risk Level", risk_level)
                with col2:
                    st.metric("Risk Probability", f"{risk_probability:.2f}%")
                with col3:
                    st.metric("Prediction Class", f"{'High (1)' if prediction_class == 1 else 'Low (0)'}")
                with col4:
                    st.metric("Model", metadata['best_model'])
                
                st.divider()
                
                # Create visualizations
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Gauge chart for risk
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_probability,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Probability", 'font': {'size': 24, 'color': 'white'}},
                        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': risk_color},
                            'bgcolor': "rgba(255,255,255,0.1)",
                            'borderwidth': 2,
                            'bordercolor': "white",
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(72, 187, 120, 0.3)'},
                                {'range': [40, 70], 'color': 'rgba(254, 225, 64, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(245, 87, 108, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_probability
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "white", 'family': "Arial"},
                        height=400
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with chart_col2:
                    # Probability breakdown
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=['Low Risk', 'High Risk'],
                            y=[prediction_proba[0]*100, prediction_proba[1]*100],
                            marker_color=['rgba(72, 187, 120, 0.8)', 'rgba(245, 87, 108, 0.8)'],
                            text=[f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"],
                            textposition='auto',
                        )
                    ])
                    
                    fig_bar.update_layout(
                        title={'text': 'Probability Breakdown', 'font': {'size': 20, 'color': 'white'}},
                        xaxis={'title': 'Risk Category', 'color': 'white'},
                        yaxis={'title': 'Probability (%)', 'color': 'white', 'range': [0, 100]},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(255,255,255,0.05)',
                        font={'color': 'white'},
                        height=400
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Feature importance (if available)
                st.divider()
                st.subheader("📊 Input Summary")
                
                summary_df = pd.DataFrame({
                    'Feature': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
    
    with tab2:
        st.header("📈 Model Performance Metrics")
        
        if metadata and 'performance' in metadata:
            # Model comparison
            models_data = []
            for model_name, metrics in metadata['performance'].items():
                models_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'] * 100,
                    'Precision': metrics['precision'] * 100,
                    'Recall': metrics['recall'] * 100,
                    'F1-Score': metrics['f1_score'] * 100,
                    'ROC-AUC': metrics['roc_auc'] * 100
                })
            
            df_models = pd.DataFrame(models_data)
            
            # Display as table
            st.dataframe(df_models, use_container_width=True, hide_index=True)
            
            # Comparison chart
            fig = px.bar(df_models, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        title='Model Performance Comparison',
                        barmode='group',
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(255,255,255,0.05)',
                font={'color': 'white'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### 🚗 Traffic Accident Risk Predictor
        
        This application uses **Machine Learning** to predict the risk level of traffic accidents based on various factors including:
        
        - **Time factors**: Hour, day, month, rush hour, weekend
        - **Weather conditions**: Rain, fog, clear weather, etc.
        - **Road characteristics**: Junction, urban/rural, highway, road type
        - **Accident details**: Number of vehicles, casualties, speed limit, driver age
        
        ### 🤖 Model Details
        
        - **Algorithm**: XGBoost Classifier
        - **Training Data**: Indian traffic accident dataset
        - **Features**: 19 input features
        - **Output**: Binary classification (High Risk / Low Risk)
        
        ### 📊 Performance
        
        The model has been trained and validated on historical traffic accident data with good accuracy metrics.
        
        ### 🎯 Use Cases
        
        - **Traffic Management**: Identify high-risk scenarios for better traffic control
        - **Insurance**: Risk assessment for insurance premiums
        - **Safety Planning**: Design safer road infrastructure
        - **Driver Awareness**: Educate drivers about risky conditions
        
        ### ⚠️ Disclaimer
        
        This is a predictive model and should be used as a supplementary tool. Always follow traffic rules and exercise caution while driving.
        """)
        
        st.divider()
        
        st.caption("Developed with ❤️ using Streamlit and Machine Learning")
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

else:
    st.error("Failed to load model. Please check if model files exist in the 'models' directory.")
