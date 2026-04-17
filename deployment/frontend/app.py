"""Streamlit frontend for SA-ZD-NIDS demo and monitoring."""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="SA-ZD-NIDS Dashboard",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">:shield: SA-ZD-NIDS Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Select Page", [
        "Demo Interface",
        "System Monitoring",
        "Drift Analysis",
        "Batch Prediction"
    ])
    
    if page == "Demo Interface":
        demo_interface()
    elif page == "System Monitoring":
        system_monitoring()
    elif page == "Drift Analysis":
        drift_analysis()
    elif page == "Batch Prediction":
        batch_prediction()

def demo_interface():
    """Interactive demo interface for single predictions."""
    st.header("Network Traffic Analysis Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Network Traffic Features")
        
        # Feature input form
        with st.form("prediction_form"):
            st.write("Enter network traffic features (example values provided):")
            
            # Create input fields for common network features
            feature_defaults = {
                "duration": 0.0,
                "src_bytes": 1000.0,
                "dst_bytes": 500.0,
                "src_packets": 10.0,
                "dst_packets": 5.0,
                "src_errors": 0.0,
                "dst_errors": 0.0,
                "service": 1.0,
                "flag": 1.0,
                "land": 0.0
            }
            
            features = {}
            for feature_name, default_value in feature_defaults.items():
                features[feature_name] = st.number_input(
                    f"{feature_name}",
                    value=default_value,
                    step=0.1,
                    key=f"feature_{feature_name}"
                )
            
            submitted = st.form_submit_button("Analyze Traffic")
            
            if submitted:
                # Prepare feature vector
                feature_values = list(features.values())
                
                # Make prediction
                with st.spinner("Analyzing network traffic..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/predict",
                            json={"features": feature_values},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            display_prediction_result(result)
                        else:
                            st.error(f"Prediction failed: {response.text}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"API connection error: {e}")
    
    with col2:
        st.subheader("Quick Info")
        
        st.info("""
        **How it works:**
        1. Enter network traffic features
        2. System analyzes traffic patterns
        3. Detects known attacks and zero-day threats
        4. Provides confidence scores and alerts
        """)
        
        st.warning("""
        **Note:** Make sure the API server is running on localhost:8000
        """)

def display_prediction_result(result):
    """Display prediction results with visual indicators."""
    st.subheader("Analysis Results")
    
    # Determine alert level based on prediction
    prediction = result["prediction"]
    confidence = result["confidence"]
    is_zero_day = result["is_zero_day"]
    
    # Alert styling
    if prediction == "ZERO_DAY" or is_zero_day:
        alert_class = "alert-high"
        alert_icon = " :rotating_light: "
        alert_text = "ZERO-DAY ATTACK DETECTED!"
    elif prediction != "BENIGN":
        alert_class = "alert-medium"
        alert_icon = " :warning: "
        alert_text = f"KNOWN ATTACK: {prediction}"
    else:
        alert_class = "alert-low"
        alert_icon = " :white_check_mark: "
        alert_text = "BENIGN TRAFFIC"
    
    # Display alert
    st.markdown(f"""
    <div class="{alert_class}">
        <h3>{alert_icon} {alert_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", prediction)
    
    with col2:
        st.metric("Confidence", f"{confidence:.3f}")
    
    with col3:
        st.metric("Zero-Day", "Yes" if is_zero_day else "No")
    
    with col4:
        st.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")
    
    # Reconstruction error for zero-day detection
    if result.get("reconstruction_error") is not None:
        st.subheader("Anomaly Analysis")
        recon_error = result["reconstruction_error"]
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = recon_error,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Reconstruction Error"},
            delta = {'reference': 0.1},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "gray"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def system_monitoring():
    """System monitoring dashboard."""
    st.header("System Performance Monitoring")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (10 seconds)")
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # Fetch system metrics
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            display_system_metrics(health_data)
        else:
            st.error(f"Failed to fetch health data: {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")

def display_system_metrics(health_data):
    """Display system performance metrics."""
    # System status
    status = health_data["status"]
    system_metrics = health_data["system_metrics"]
    drift_metrics = health_data["drift_metrics"]
    
    # Status indicator
    if status == "healthy":
        st.success(f"System Status: {status.upper()}")
    else:
        st.error(f"System Status: {status.upper()}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU Usage", f"{system_metrics['cpu_usage']:.1f}%")
    
    with col2:
        st.metric("Memory Usage", f"{system_metrics['memory_usage_mb']:.1f} MB")
    
    with col3:
        st.metric("Avg Latency", f"{system_metrics['avg_latency_ms']:.2f} ms")
    
    with col4:
        st.metric("Requests/min", f"{system_metrics['requests_per_minute']:.1f}")
    
    # Drift information
    st.subheader("Drift Detection Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drift_status = "DETECTED" if drift_metrics["drift_detected"] else "Normal"
        if drift_metrics["drift_detected"]:
            st.error(f"Drift: {drift_status}")
        else:
            st.success(f"Drift: {drift_status}")
    
    with col2:
        st.metric("Drift Score", f"{drift_metrics['drift_score']:.3f}")
    
    with col3:
        st.metric("Total Drifts", drift_metrics["num_drifts_total"])
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Version", system_metrics["model_version"])
    
    with col2:
        uptime_hours = system_metrics["uptime_seconds"] / 3600
        st.metric("Uptime", f"{uptime_hours:.1f} hours")

def drift_analysis():
    """Drift analysis and visualization."""
    st.header("Concept Drift Analysis")
    
    # Fetch drift statistics
    try:
        response = requests.get(f"{API_BASE_URL}/predict/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            display_drift_analysis(stats)
        else:
            st.error(f"Failed to fetch stats: {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")

def display_drift_analysis(stats):
    """Display drift analysis charts."""
    # Create sample drift data for visualization
    # In production, this would come from the API
    sample_drift_data = generate_sample_drift_data()
    
    st.subheader("Drift Score Over Time")
    
    fig = go.Figure()
    
    # Add drift score line
    fig.add_trace(go.Scatter(
        x=sample_drift_data["timestamp"],
        y=sample_drift_data["drift_score"],
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='blue', width=2)
    ))
    
    # Add threshold line
    fig.add_trace(go.Scatter(
        x=sample_drift_data["timestamp"],
        y=[0.5] * len(sample_drift_data["timestamp"]),
        mode='lines',
        name='Drift Threshold',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Mark drift points
    drift_points = sample_drift_data[sample_drift_data["drift_detected"]]
    if len(drift_points) > 0:
        fig.add_trace(go.Scatter(
            x=drift_points["timestamp"],
            y=drift_points["drift_score"],
            mode='markers',
            name='Drift Detected',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title="Concept Drift Detection",
        xaxis_title="Time",
        yaxis_title="Drift Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Drift Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", stats["total_predictions"])
    
    with col2:
        st.metric("Zero-Day Detections", stats["zero_day_detections"])
    
    with col3:
        zd_rate = (stats["zero_day_detections"] / max(1, stats["total_predictions"])) * 100
        st.metric("Zero-Day Rate", f"{zd_rate:.2f}%")

def batch_prediction():
    """Batch prediction interface."""
    st.header("Batch Prediction")
    
    # File upload or sample data
    st.subheader("Upload Network Traffic Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with network traffic features"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display sample data
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Select features for prediction
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Convert DataFrame to feature list
                    feature_columns = [col for col in df.columns if col != 'timestamp']
                    batch_features = df[feature_columns].values.tolist()
                    
                    # Make batch prediction
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/predict/batch",
                            json={"batch_features": batch_features},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            display_batch_results(df, result)
                        else:
                            st.error(f"Batch prediction failed: {response.text}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"API connection error: {e}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        # Provide sample data option
        st.info("Or use sample data for demonstration:")
        if st.button("Generate Sample Data"):
            sample_df = generate_sample_network_data()
            st.dataframe(sample_df)
            
            if st.button("Predict Sample Data"):
                batch_features = sample_df.iloc[:, :-1].values.tolist()
                
                with st.spinner("Processing sample predictions..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/predict/batch",
                            json={"batch_features": batch_features},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            display_batch_results(sample_df, result)
                        else:
                            st.error(f"Batch prediction failed: {response.text}")
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"API connection error: {e}")

def display_batch_results(original_df, results):
    """Display batch prediction results."""
    st.subheader("Batch Prediction Results")
    
    # Create results DataFrame
    results_df = original_df.copy()
    results_df['prediction'] = results['predictions']
    results_df['confidence'] = results['confidences']
    results_df['is_zero_day'] = results['zero_day_flags']
    results_df['reconstruction_error'] = results['reconstruction_errors']
    
    # Display results
    st.dataframe(results_df)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = len(results_df)
    zero_days = results_df['is_zero_day'].sum()
    attacks = (results_df['prediction'] != 'BENIGN').sum() - zero_days
    benign = total_samples - attacks - zero_days
    
    with col1:
        st.metric("Total Samples", total_samples)
    
    with col2:
        st.metric("Benign", benign)
    
    with col3:
        st.metric("Known Attacks", attacks)
    
    with col4:
        st.metric("Zero-Day", zero_days)
    
    # Visualization
    st.subheader("Prediction Distribution")
    
    # Pie chart
    fig = px.pie(
        values=[benign, attacks, zero_days],
        names=['Benign', 'Known Attacks', 'Zero-Day'],
        title='Traffic Classification Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    st.subheader("Confidence Score Distribution")
    
    fig = px.histogram(
        results_df,
        x='confidence',
        title='Confidence Score Distribution',
        nbins=20
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_sample_drift_data():
    """Generate sample drift data for visualization."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='1H'
    )
    
    drift_scores = np.random.normal(0.2, 0.1, len(timestamps))
    
    # Add some drift events
    drift_indices = [5, 12, 18]
    for idx in drift_indices:
        if idx < len(drift_scores):
            drift_scores[idx] = np.random.normal(0.8, 0.1)
    
    drift_detected = drift_scores > 0.5
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'drift_score': drift_scores,
        'drift_detected': drift_detected
    })

def generate_sample_network_data():
    """Generate sample network traffic data."""
    np.random.seed(42)
    
    # Generate benign traffic
    benign_samples = 50
    benign_data = {
        'duration': np.random.exponential(0.1, benign_samples),
        'src_bytes': np.random.lognormal(6, 1, benign_samples),
        'dst_bytes': np.random.lognormal(5, 1, benign_samples),
        'src_packets': np.random.poisson(10, benign_samples),
        'dst_packets': np.random.poisson(5, benign_samples),
        'src_errors': np.random.poisson(0, benign_samples),
        'dst_errors': np.random.poisson(0, benign_samples),
        'service': np.random.randint(0, 5, benign_samples),
        'flag': np.random.randint(0, 3, benign_samples),
        'land': np.zeros(benign_samples)
    }
    
    # Generate attack traffic
    attack_samples = 30
    attack_data = {
        'duration': np.random.exponential(1.0, attack_samples),
        'src_bytes': np.random.lognormal(8, 2, attack_samples),
        'dst_bytes': np.random.lognormal(7, 2, attack_samples),
        'src_packets': np.random.poisson(50, attack_samples),
        'dst_packets': np.random.poisson(25, attack_samples),
        'src_errors': np.random.poisson(2, attack_samples),
        'dst_errors': np.random.poisson(1, attack_samples),
        'service': np.random.randint(0, 5, attack_samples),
        'flag': np.random.randint(0, 3, attack_samples),
        'land': np.random.randint(0, 2, attack_samples)
    }
    
    # Combine data
    all_data = {}
    for key in benign_data:
        all_data[key] = list(benign_data[key]) + list(attack_data[key])
    
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    main()
