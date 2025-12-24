import streamlit as st
import pandas as pd
from joblib import load
import os

# Check if model exists
if not os.path.exists('churn_model.joblib'):
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return load('churn_model.joblib')

model = load_model()

# Prediction function
def predict_churn(inputs):
    # Payment method encoding
    payment_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer': 2,
        'Credit card': 3
    }
    
    # Create DataFrame with all expected columns
    input_df = pd.DataFrame([{
        'gender': 1 if inputs['gender'] == 'Male' else 0,
        'SeniorCitizen': 0,  # default
        'Partner': 0,        # default
        'Dependents': 0,     # default
        'tenure': inputs['tenure'],
        'PhoneService': 1,   # default
        'MultipleLines': 0,  # default
        'InternetService': 1 if inputs['internet_service'] == 'Yes' else 0,
        'OnlineSecurity': 1 if inputs['online_security'] == 'Yes' else 0,
        'OnlineBackup': 0,   # default
        'DeviceProtection': 0,# default
        'TechSupport': 0,    # default
        'StreamingTV': 0,    # default
        'StreamingMovies': 0, # default
        'Contract': 0 if inputs['contract'] == 'Month-to-month' else (1 if inputs['contract'] == 'One year' else 2),
        'PaperlessBilling': 0,# default
        'PaymentMethod': payment_map[inputs['payment_method']],
        'MonthlyCharges': inputs['monthly_charges'],
        'TotalCharges': inputs['tenure'] * inputs['monthly_charges']
    }])
    
    proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    return proba, prediction

# Streamlit app
def main():
    st.title("Customer Churn Prediction")
    
    with st.form("prediction_form"):
        st.subheader("Customer Details")
        
        # Essential inputs + requested additions
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 200.0, 70.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer", "Credit card"])
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            inputs = {
                'gender': gender,
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'contract': contract,
                'internet_service': internet_service,
                'online_security': online_security,
                'payment_method': payment_method
            }
            
            proba, prediction = predict_churn(inputs)
            
            # Display results
            st.subheader("Results")
            st.metric("Churn Probability", f"{proba:.1%}")
            
            if prediction == 1:
                st.error("High churn risk detected")
                st.write("Recommended action: Offer retention discount")
            else:
                st.success("Low churn risk")
                st.write("No immediate action needed")
            
            # Simple summary
            st.write("Based on:")
            st.write(f"- Gender: {gender}")
            st.write(f"- Tenure: {tenure} months")
            st.write(f"- Monthly Charges: ${monthly_charges:.2f}")
            st.write(f"- Contract: {contract}")
            st.write(f"- Internet Service: {internet_service}")
            st.write(f"- Online Security: {online_security}")
            st.write(f"- Payment Method: {payment_method}")

if __name__ == "__main__":
    main()
