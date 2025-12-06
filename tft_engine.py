import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
import requests
from io import StringIO

# Configure page
st.set_page_config(
    page_title="TFT-Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem !important;
    text-align: center;
    color: #1E40AF;
    margin-bottom: 10px;
    font-weight: bold;
}
.sub-header {
    font-size: 1.3rem !important;
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}
.input-section {
    font-size: 1.3rem !important;
    text-align: center;
    color: #333;
    margin: 30px 0 20px 0;
    text-decoration: underline;
    font-weight: bold;
}
.stSelectbox label, .stNumberInput label {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #333 !important;
}
.stButton button {
    font-size: 1.3rem !important;
    font-weight: bold !important;
    padding: 0.6rem 2.5rem !important;
    background-color: #1E40AF !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
}
.result-container {
    background-color: #f0f2f6;
    padding: 25px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
    border: 3px solid #ddd;
}
.result-text {
    font-size: 1.6rem !important;
    font-weight: bold !important;
    color: #333;
}
.high-quality {
    color: #00b894 !important;
    background-color: #e6f7f0 !important;
    border-color: #00b894 !important;
}
.medium-quality {
    color: #f39c12 !important;
    background-color: #fff7e6 !important;
    border-color: #f39c12 !important;
}
.low-quality {
    color: #d63031 !important;
    background-color: #ffe6e6 !important;
    border-color: #d63031 !important;
}
.footer {
    text-align: center;
    font-size: 1rem;
    color: #666;
    margin-top: 40px;
    padding: 20px;
    border-top: 2px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_train_model():
    """Load dataset, preprocess, and train XGBoost model"""
    try:
        # Load data from Google Sheets
        csv_url = "https://docs.google.com/spreadsheets/d/16Bup9C6-JiWqzMb1sMJafYOsmIO76SFNEdiK9Tr96do/export?format=csv"
        
        response = requests.get(csv_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
        else:
            st.error("Could not load data from Google Sheets")
            return None, None, None, None
        
        # Separate features and target
        X = df.drop('Device_Quality', axis=1)
        y = df['Device_Quality']
        
        # ============================================================================
        # ONE-HOT ENCODING (BEFORE OUTLIER REMOVAL)
        # ============================================================================
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # One-Hot Encoding for categorical features
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        
        # Clean column names for compatibility
        X_encoded.columns = X_encoded.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        
        # Combine for outlier removal
        df_encoded = X_encoded.copy()
        df_encoded['Device_Quality'] = y.values
        
        # ============================================================================
        # OUTLIER REMOVAL USING IQR (INTERQUARTILE RANGE)
        # ============================================================================
        
        # Identify numerical columns for outlier removal
        numerical_cols_for_outlier_detection = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate IQR and remove outliers
        outlier_indices = set()
        
        for col in numerical_cols_for_outlier_detection:
            Q1 = df_encoded[col].quantile(0.25)
            Q3 = df_encoded[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outlier indices
            col_outliers = df_encoded[(df_encoded[col] < lower_bound) | (df_encoded[col] > upper_bound)].index
            outlier_indices.update(col_outliers)
        
        # Remove outliers
        df_clean = df_encoded.drop(index=list(outlier_indices))
        df_clean = df_clean.reset_index(drop=True)
        
        # Update X and y with cleaned data
        X_clean = df_clean.drop('Device_Quality', axis=1)
        y_clean = df_clean['Device_Quality']
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_clean)
        
        # ============================================================================
        # STANDARD SCALING
        # ============================================================================
        
        # Initialize StandardScaler
        scaler = StandardScaler()
        
        # Fit and transform the encoded features
        X_scaled = scaler.fit_transform(X_clean)
        
        # Convert back to DataFrame to preserve column names
        X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
        
        # ============================================================================
        # HANDLE IMBALANCED DATA - Random Oversampling
        # ============================================================================
        
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_scaled, y_encoded)
        
        # ============================================================================
        # TRAIN XGBOOST MODEL
        # ============================================================================
        
        # Train XGBoost on entire resampled dataset
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_resampled, y_resampled)
        
        # Store original categorical columns info
        categorical_info = {
            'Substrate_Type': df['Substrate_Type'].unique().tolist(),
            'Deposition_Method': df['Deposition_Method'].unique().tolist(),
            'Material_Type': df['Material_Type'].unique().tolist()
        }
        
        return xgb_model, scaler, le, X_clean.columns.tolist(), categorical_info
        
    except Exception as e:
        st.error(f"Error loading data or training model: {str(e)}")
        return None, None, None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">TFT-Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nano-structured Metal-Oxide Thin-Film Transistor Device Quality Predictor</p>', unsafe_allow_html=True)
    
    # Load model and data
    with st.spinner('Loading model and data...'):
        model, scaler, label_encoder, feature_columns, categorical_info = load_and_train_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the data source.")
        return
    
    # Input section header
    st.markdown('<p class="input-section">Input Parameters</p>', unsafe_allow_html=True)
    
    # Input form in 4 rows, 5 columns layout
    # Row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        substrate_type = st.selectbox(
            "Substrate Type",
            options=categorical_info['Substrate_Type'],
            index=0
        )
    
    with col2:
        deposition_method = st.selectbox(
            "Deposition Method",
            options=categorical_info['Deposition_Method'],
            index=0
        )
    
    with col3:
        material_type = st.selectbox(
            "Material Type",
            options=categorical_info['Material_Type'],
            index=0
        )
    
    with col4:
        film_thickness = st.number_input(
            "Film Thickness (nm)",
            min_value=20.02,
            max_value=149.96,
            value=85.49,
            step=0.1
        )
    
    with col5:
        annealing_temp = st.number_input(
            "Annealing Temp (°C)",
            min_value=150.09,
            max_value=349.91,
            value=252.90,
            step=0.1
        )
    
    # Row 2
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        nanostructure_size = st.number_input(
            "Nanostructure Size (nm)",
            min_value=5.01,
            max_value=79.82,
            value=41.10,
            step=0.1
        )
    
    with col2:
        gate_length = st.number_input(
            "Gate Length (μm)",
            min_value=0.52,
            max_value=9.99,
            value=5.15,
            step=0.01
        )
    
    with col3:
        gate_width = st.number_input(
            "Gate Width (μm)",
            min_value=10.12,
            max_value=199.97,
            value=104.58,
            step=0.1
        )
    
    with col4:
        mobility = st.number_input(
            "Mobility (cm²V⁻¹s⁻¹)",
            min_value=5.16,
            max_value=59.90,
            value=33.24,
            step=0.01
        )
    
    with col5:
        threshold_voltage = st.number_input(
            "Threshold Voltage (V)",
            min_value=-1.49,
            max_value=2.99,
            value=0.76,
            step=0.01
        )
    
    # Row 3
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        on_off_ratio = st.number_input(
            "On/Off Ratio",
            min_value=123136,
            max_value=99949640,
            value=49174980,
            step=1000
        )
    
    with col2:
        subthreshold_swing = st.number_input(
            "Subthreshold Swing (V/dec)",
            min_value=0.05,
            max_value=0.50,
            value=0.28,
            step=0.01
        )
    
    with col3:
        gate_density = st.number_input(
            "Gate Density (gates/mm²)",
            min_value=102.07,
            max_value=1998.73,
            value=1043.32,
            step=0.1
        )
    
    with col4:
        bending_radius = st.number_input(
            "Bending Radius (mm)",
            min_value=2.00,
            max_value=29.97,
            value=16.05,
            step=0.01
        )
    
    with col5:
        cycles_to_failure = st.number_input(
            "Cycles to Failure",
            min_value=1169,
            max_value=49976,
            value=25210,
            step=1
        )
    
    # Row 4
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        power_consumption = st.number_input(
            "Power Consumption (mW)",
            min_value=0.10,
            max_value=10.00,
            value=5.06,
            step=0.01
        )
    
    with col2:
        snn_accuracy = st.number_input(
            "SNN Accuracy (%)",
            min_value=70.26,
            max_value=98.95,
            value=84.58,
            step=0.01
        )
    
    with col3:
        response_time = st.number_input(
            "Response Time (ms)",
            min_value=0.10,
            max_value=4.99,
            value=2.54,
            step=0.01
        )
    
    with col4:
        temperature_stability = st.number_input(
            "Temperature Stability (%)",
            min_value=85.01,
            max_value=99.95,
            value=92.22,
            step=0.01
        )
    
    with col5:
        transparency = st.number_input(
            "Transparency (%)",
            min_value=60.03,
            max_value=94.99,
            value=77.36,
            step=0.01
        )
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        submit_button = st.button("Submit", type="primary")
    
    # Prediction and results
    if submit_button:
        try:
            # Create input dataframe with original feature names
            input_data = pd.DataFrame({
                'Substrate_Type': [substrate_type],
                'Deposition_Method': [deposition_method],
                'Material_Type': [material_type],
                'Film_Thickness_nm': [film_thickness],
                'Annealing_Temp_C': [annealing_temp],
                'Nanostructure_Size_nm': [nanostructure_size],
                'Gate_Length_um': [gate_length],
                'Gate_Width_um': [gate_width],
                'Mobility_cm2V-1s-1': [mobility],
                'Threshold_Voltage_V': [threshold_voltage],
                'On_Off_Ratio': [on_off_ratio],
                'Subthreshold_Swing_Vdec': [subthreshold_swing],
                'Gate_Density_gates_per_mm2': [gate_density],
                'Bending_Radius_mm': [bending_radius],
                'Cycles_to_Failure': [cycles_to_failure],
                'Power_Consumption_mW': [power_consumption],
                'SNN_Accuracy_%': [snn_accuracy],
                'Response_Time_ms': [response_time],
                'Temperature_Stability_%': [temperature_stability],
                'Transparency_%': [transparency]
            })
            
            # One-hot encode categorical variables
            categorical_cols = ['Substrate_Type', 'Deposition_Method', 'Material_Type']
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=False)
            
            # Clean column names
            input_encoded.columns = input_encoded.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
            
            # Ensure all training features are present
            for col in feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reorder columns to match training data
            input_encoded = input_encoded[feature_columns]
            
            # Scale the input
            input_scaled = scaler.transform(input_encoded)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Decode prediction
            device_quality = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            
            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if device_quality == "High":
                result_class = "high-quality"
            elif device_quality == "Medium":
                result_class = "medium-quality"
            else:
                result_class = "low-quality"
            
            st.markdown(f"""
            <div class="result-container {result_class}">
                <p class="result-text">The Device Quality is</p>
                <h2 style="margin: 15px 0; font-size: 3rem;">{device_quality}</h2>
                <p style="font-size: 1.3rem; margin-top: 15px;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    
    else:
        # Show empty result container
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-container">
            <p class="result-text">The Device Quality is</p>
            <p style="font-size: 1.2rem; color: #666; margin-top: 15px;">Click "Predict Device Quality" to see results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>The XGBoost-Based TFT Device Quality Prediction Web Application is Developed By Authors</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()