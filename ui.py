import streamlit as st
import requests
import pandas as pd

st.title("NASA Turbofan RUL Predictor")

# 1. UI component to upload a CSV snippet
uploaded_file = st.file_uploader("Upload continuous sensor logs (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Input Preview", df.tail(3))
    
    # Convert dataframe row or sequence to JSON payload matching your FastAPI schema
    payload = df.to_dict(orient="records") 

    if st.button("Analyze Engine Health"):
        # 2. Fire a POST request to your FastAPI backend
        # (Replace localhost with your production URL once deployed)
        backend_url = "http://127.0.0.1:8000/predict" 
        
        with st.spinner("Querying LSTM model backend..."):
            response = requests.post(backend_url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            st.success(f"Estimated Remaining Useful Life: **{result['rul']} Cycles**")
        else:
            st.error("Failed to fetch prediction from backend application.")