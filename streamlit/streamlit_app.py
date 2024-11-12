import streamlit as st
import pandas as pd
import tomli
import requests
from pathlib import Path
import os

# API configuration
API_URL = os.getenv('API_URL', 'http://api:8000')

def load_config():
    """Load configuration from TOML file."""
    with open('config.toml', 'rb') as f:
        return tomli.load(f)

def initialize_session_state(config):
    """Initialize session state with default values."""
    # Initialize values for group 1 sensors
    for feature in config['sensor_groups']['group1']:
        if f"group1_{feature}" not in st.session_state:
            st.session_state[f"group1_{feature}"] = float(config['feature_ranges'][feature]['default'])

    # Initialize values for group 2 sensors
    for feature in config['sensor_groups']['group2']:
        if f"group2_{feature}" not in st.session_state:
            st.session_state[f"group2_{feature}"] = float(config['feature_ranges'][feature]['default'])

    # Initialize preset selector if not exists
    if 'preset_selector' not in st.session_state:
        st.session_state.preset_selector = "Custom"

def create_sensor_input(feature, range_info, key):
    """Create a standardized sensor input widget with additional information."""
    col1, col2 = st.columns([4, 1])
    
    display_name = feature.replace('_', ' ').title()
    
    with col1:
        value = st.slider(
            display_name,
            min_value=float(range_info['min']),
            max_value=float(range_info['max']),
            value=st.session_state[key],
            key=key,
            on_change=None  # Remove default on_change handler
        )
    
    with col2:
        st.write(f"Value: {value:.2f}")
    
    return value

def get_prediction(sensor_values):
    """Make API call to get prediction."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"sensor_values": sensor_values},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to prediction service: {str(e)}")
        return None

def update_preset_values():
    """Update sensor values based on preset selection."""
    preset = st.session_state.preset_selector
    config = st.session_state.config
    
    if preset == "Normal Operation":
        values = {feature: range_info['default'] 
                 for feature, range_info in config['feature_ranges'].items()}
    elif preset == "High Risk":
        values = {}
        for feature, range_info in config['feature_ranges'].items():
            value_range = range_info['max'] - range_info['min']
            values[feature] = range_info['max'] - (value_range * 0.2)
    else:
        return

    # Update session state values
    for feature in config['sensor_groups']['group1']:
        st.session_state[f"group1_{feature}"] = values[feature]
    for feature in config['sensor_groups']['group2']:
        st.session_state[f"group2_{feature}"] = values[feature]

def main():
    # Load and store configuration in session state
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    # Initialize session state
    initialize_session_state(st.session_state.config)
    
    # Page configuration
    st.set_page_config(
        page_title="Engine Maintenance Risk Predictor",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stSlider {margin-bottom: 2rem;}
        .risk-header {font-size: 1.5rem; font-weight: bold;}
        .sensor-group {background-color: #1E1E1E; padding: 1rem; border-radius: 0.5rem;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔧 Engine Maintenance Risk Predictor")
    
    # Create main columns
    col1, col2, col3 = st.columns([2, 2, 3])
    
    # Sensors setup
    with col1:
        st.markdown('<p class="risk-header">Sensors Setup</p>', unsafe_allow_html=True)
        with st.container():
            group1_values = {}
            for feature in st.session_state.config['sensor_groups']['group1']:
                range_info = st.session_state.config['feature_ranges'][feature]
                group1_values[feature] = create_sensor_input(
                    feature, range_info, f"group1_{feature}"
                )
    
    with col2:
        st.markdown('<div style="height: 3.5rem;"></div>', unsafe_allow_html=True)
        with st.container():
            group2_values = {}
            for feature in st.session_state.config['sensor_groups']['group2']:
                range_info = st.session_state.config['feature_ranges'][feature]
                group2_values[feature] = create_sensor_input(
                    feature, range_info, f"group2_{feature}"
                )
    
    # Prediction column
    with col3:
        st.markdown('<p class="risk-header">Prediction</p>', unsafe_allow_html=True)
        
        # Preset selector with callback
        st.selectbox(
            "Select Preset",
            options=["Custom", "Normal Operation", "High Risk"],
            key="preset_selector",
            on_change=update_preset_values
        )
        
        if st.button("Predict Risk Level", type="primary", key="predict"):
            # Combine all input values
            input_values = {**group1_values, **group2_values}
            
            # Get prediction from API
            prediction_result = get_prediction(input_values)
            
            if prediction_result:
                # Display results
                st.markdown(f"### Predicted Risk Level:")
                st.markdown(
                    f"<h2 style='color: {prediction_result['color']}'>{prediction_result['risk_label']}</h2>", 
                    unsafe_allow_html=True
                )
                
                # Display probability distribution
                st.write("### Risk Probabilities:")
                risk_labels = [
                    ("Low Risk (>30 cycles)", "green"),
                    ("Medium Risk (20-30 cycles)", "yellow"),
                    ("High Risk (10-20 cycles)", "orange"),
                    ("Higher Risk (0-10 cycles)", "red")
                ]
                
                for (label, color), prob in zip(risk_labels, prediction_result['risk_probabilities']):
                    st.markdown(
                        f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
                        f"<div style='flex-grow: 1; margin-right: 10px;'>"
                        f"<div style='background-color: {color}; "
                        f"width: {prob*100}%; height: 20px; border-radius: 10px;'></div>"
                        f"</div>"
                        f"<div style='width: 150px;'>{label}: {prob:.1%}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()