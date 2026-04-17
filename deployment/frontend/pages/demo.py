"""Demo page for SA-ZD-NIDS interactive demonstration."""
import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def demo_page():
    """Main demo page with interactive workflow."""
    st.title("SA-ZD-NIDS Interactive Demo")
    st.markdown("---")
    
    # Demo workflow steps
    st.header("Network Traffic Analysis Workflow")
    
    # Step 1: Input
    st.subheader("Step 1: Input Network Traffic Data")
    
    input_method = st.radio("Choose input method:", ["Manual Input", "Sample Data", "Upload File"])
    
    if input_method == "Manual Input":
        features = manual_input_interface()
    elif input_method == "Sample Data":
        features = sample_data_interface()
    else:
        features = file_upload_interface()
    
    if features is None:
        st.warning("Please provide network traffic features to continue.")
        return
    
    # Step 2: Analysis
    st.subheader("Step 2: Analyze Traffic Patterns")
    
    if st.button("Analyze Traffic", type="primary", use_container_width=True):
        with st.spinner("Analyzing network traffic..."):
            # Make prediction
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json={"features": features},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Step 3: Results
                    st.subheader("Step 3: Analysis Results")
                    display_detailed_results(result, features)
                    
                    # Step 4: Explanation
                    st.subheader("Step 4: Threat Analysis")
                    display_threat_analysis(result, features)
                    
                else:
                    st.error(f"Analysis failed: {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

def manual_input_interface():
    """Manual feature input interface."""
    st.write("Enter network traffic features:")
    
    # Feature categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Basic Traffic Info**")
        duration = st.number_input("Duration (seconds)", value=0.0, min_value=0.0, step=0.1)
        src_bytes = st.number_input("Source Bytes", value=1000.0, min_value=0.0, step=100.0)
        dst_bytes = st.number_input("Destination Bytes", value=500.0, min_value=0.0, step=100.0)
    
    with col2:
        st.write("**Packet Information**")
        src_packets = st.number_input("Source Packets", value=10.0, min_value=0.0, step=1.0)
        dst_packets = st.number_input("Destination Packets", value=5.0, min_value=0.0, step=1.0)
        src_errors = st.number_input("Source Errors", value=0.0, min_value=0.0, step=1.0)
    
    with col3:
        st.write("**Connection Details**")
        dst_errors = st.number_input("Destination Errors", value=0.0, min_value=0.0, step=1.0)
        service = st.selectbox("Service Type", [0, 1, 2, 3, 4], format_func=lambda x: f"Service {x}")
        flag = st.selectbox("Flag Status", [0, 1, 2], format_func=lambda x: ["Normal", "Warning", "Alert"][x])
        land = st.selectbox("Land Attack", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    
    # Additional features
    with st.expander("Advanced Features"):
        col1, col2 = st.columns(2)
        with col1:
            urgent = st.number_input("Urgent Packets", value=0.0, min_value=0.0, step=1.0)
            hot = st.number_input("Hot Indicators", value=0.0, min_value=0.0, step=1.0)
        with col2:
            failed_logins = st.number_input("Failed Logins", value=0.0, min_value=0.0, step=1.0)
            num_compromised = st.number_input("Compromised Systems", value=0.0, min_value=0.0, step=1.0)
    
    # Compile features
    features = [
        duration, src_bytes, dst_bytes, src_packets, dst_packets,
        src_errors, dst_errors, service, flag, land,
        urgent, hot, failed_logins, num_compromised
    ]
    
    return features

def sample_data_interface():
    """Sample data interface."""
    st.write("Select sample traffic type:")
    
    sample_type = st.selectbox(
        "Traffic Type",
        ["Benign Traffic", "DDoS Attack", "Port Scan", "Zero-Day Attack"],
        help="Choose a pre-configured sample for demonstration"
    )
    
    # Sample feature vectors
    sample_features = {
        "Benign Traffic": [0.1, 1000.0, 500.0, 10.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "DDoS Attack": [5.0, 10000.0, 1000.0, 100.0, 50.0, 10.0, 5.0, 2.0, 1.0, 0.0, 20.0, 10.0, 5.0, 2.0],
        "Port Scan": [0.01, 100.0, 50.0, 200.0, 100.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        "Zero-Day Attack": [2.0, 5000.0, 2000.0, 80.0, 40.0, 15.0, 8.0, 4.0, 2.0, 0.0, 10.0, 15.0, 3.0, 5.0]
    }
    
    selected_features = sample_features[sample_type]
    
    # Display features in a table
    feature_names = [
        "Duration", "Src Bytes", "Dst Bytes", "Src Packets", "Dst Packets",
        "Src Errors", "Dst Errors", "Service", "Flag", "Land",
        "Urgent", "Hot", "Failed Logins", "Compromised"
    ]
    
    df = pd.DataFrame({
        "Feature": feature_names,
        "Value": selected_features
    })
    
    st.dataframe(df, use_container_width=True)
    
    return selected_features

def file_upload_interface():
    """File upload interface."""
    uploaded_file = st.file_uploader(
        "Upload CSV file with features",
        type=['csv'],
        help="File should contain feature columns in the same order as manual input"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Show first few rows
            st.dataframe(df.head())
            
            # Use first row for demo
            if len(df) > 0:
                features = df.iloc[0].values.tolist()
                st.info(f"Using first row for demonstration. Features: {len(features)}")
                return features
            else:
                st.error("File is empty")
                return None
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    
    return None

def display_detailed_results(result, features):
    """Display detailed prediction results."""
    prediction = result["prediction"]
    confidence = result["confidence"]
    is_zero_day = result["is_zero_day"]
    reconstruction_error = result.get("reconstruction_error")
    processing_time = result["processing_time_ms"]
    
    # Alert styling
    if prediction == "ZERO_DAY" or is_zero_day:
        alert_type = "error"
        alert_icon = " :rotating_light: "
        alert_title = "ZERO-DAY ATTACK DETECTED!"
        alert_message = "Unknown attack pattern detected. Immediate investigation required."
    elif prediction != "BENIGN":
        alert_type = "warning"
        alert_icon = " :warning: "
        alert_title = f"KNOWN ATTACK: {prediction}"
        alert_message = "Recognized attack pattern detected. Standard protocols apply."
    else:
        alert_type = "success"
        alert_icon = " :white_check_mark: "
        alert_title = "BENIGN TRAFFIC"
        alert_message = "Normal network traffic detected. No action required."
    
    # Display alert
    if alert_type == "error":
        st.error(f"{alert_icon} {alert_title}")
        st.error(alert_message)
    elif alert_type == "warning":
        st.warning(f"{alert_icon} {alert_title}")
        st.warning(alert_message)
    else:
        st.success(f"{alert_icon} {alert_title}")
        st.success(alert_message)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", prediction)
    
    with col2:
        st.metric("Confidence", f"{confidence:.3f}")
    
    with col3:
        st.metric("Zero-Day", "Yes" if is_zero_day else "No")
    
    with col4:
        st.metric("Processing Time", f"{processing_time:.2f} ms")
    
    # Reconstruction error analysis
    if reconstruction_error is not None:
        st.subheader("Anomaly Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge chart for reconstruction error
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = reconstruction_error,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Reconstruction Error"},
                delta = {'reference': 0.1},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
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
        
        with col2:
            st.write("**Error Interpretation:**")
            if reconstruction_error < 0.3:
                st.success("Low error")
                st.write("Normal pattern")
            elif reconstruction_error < 0.7:
                st.warning("Medium error")
                st.write("Suspicious pattern")
            else:
                st.error("High error")
                st.write("Anomalous pattern")

def display_threat_analysis(result, features):
    """Display detailed threat analysis."""
    prediction = result["prediction"]
    confidence = result["confidence"]
    is_zero_day = result["is_zero_day"]
    
    # Threat assessment
    st.write("**Threat Assessment:**")
    
    if is_zero_day:
        threat_level = "CRITICAL"
        threat_color = "red"
        threat_description = """
        **Zero-Day Attack Detected**
        - Unknown attack pattern
        - No signature available
        - Immediate investigation required
        - Consider blocking source IP
        - Monitor for similar patterns
        """
    elif prediction != "BENIGN":
        threat_level = "HIGH"
        threat_color = "orange"
        threat_description = f"""
        **Known Attack: {prediction}**
        - Recognized attack pattern
        - Apply standard mitigation
        - Check IDS signatures
        - Monitor for escalation
        """
    else:
        threat_level = "LOW"
        threat_color = "green"
        threat_description = """
        **Benign Traffic**
        - Normal network activity
        - No immediate threats
        - Continue monitoring
        """
    
    # Display threat level
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f'<div style="background-color: {threat_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;"><h3>{threat_level}</h3></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(threat_description)
    
    # Feature analysis
    st.write("**Feature Analysis:**")
    
    feature_names = [
        "Duration", "Src Bytes", "Dst Bytes", "Src Packets", "Dst Packets",
        "Src Errors", "Dst Errors", "Service", "Flag", "Land",
        "Urgent", "Hot", "Failed Logins", "Compromised"
    ]
    
    # Create feature importance visualization
    feature_importance = np.abs(features)  # Simple importance based on magnitude
    feature_importance = feature_importance / np.max(feature_importance)  # Normalize
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=feature_importance,
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Feature Contribution Analysis",
        xaxis_title="Features",
        yaxis_title="Normalized Importance",
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.write("**Security Recommendations:**")
    
    if is_zero_day:
        recommendations = [
            "Immediately isolate affected systems",
            "Capture network traffic for analysis",
            "Block source IP addresses",
            "Update security rules",
            "Contact security team"
        ]
    elif prediction != "BENIGN":
        recommendations = [
            "Apply appropriate security patches",
            "Update firewall rules",
            "Monitor for related activities",
            "Review access logs",
            "Update threat intelligence"
        ]
    else:
        recommendations = [
            "Continue normal monitoring",
            "Maintain security posture",
            "Regular system updates",
            "Periodic security audits",
            "User training refreshers"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Historical context
    st.write("**Historical Context:**")
    
    # Sample historical data (in production, this would come from database)
    historical_data = {
        "Time Period": ["Last Hour", "Last 24h", "Last Week", "Last Month"],
        "Similar Threats": [0, 2, 5, 12],
        "Zero-Day Threats": [1 if is_zero_day else 0, 3, 8, 15]
    }
    
    df_historical = pd.DataFrame(historical_data)
    st.dataframe(df_historical, use_container_width=True)
    
    # Export results
    st.write("**Export Results:**")
    
    if st.button("Export Analysis Report"):
        # Create report
        report = {
            "timestamp": result["timestamp"],
            "prediction": prediction,
            "confidence": confidence,
            "is_zero_day": is_zero_day,
            "threat_level": threat_level,
            "features": dict(zip(feature_names, features)),
            "reconstruction_error": result.get("reconstruction_error"),
            "processing_time_ms": result["processing_time_ms"]
        }
        
        # Convert to JSON for download
        import json
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"nids_analysis_{int(time.time())}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    demo_page()
