import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib

# Load the scaler and model
scaler = joblib.load("my_scaler.save")
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
def main():
    st.title("Water Quality Prediction")

    st.markdown("""
    <style>
    .title {font-size:60px; font-weight: bold;}
    .safe {color: green;}
    .not-safe {color: red;}
    </style>
    """, unsafe_allow_html=True)
    
    # Input fields
    st.sidebar.header("Enter Water Quality Parameters")
    
    ph = st.sidebar.number_input("pH value", format="%.2f")
    hardness = st.sidebar.number_input("Hardness", format="%.2f")
    solids = st.sidebar.number_input("Solids", format="%.2f")
    sulfate = st.sidebar.number_input("Sulfate", format="%.2f")
    organic_carbon = st.sidebar.number_input("Organic carbon", format="%.2f")
    chloramines = st.sidebar.number_input("Chloramines", format="%.2f")
    conductivity = st.sidebar.number_input("Conductivity", format="%.2f")
    trihalomethanes = st.sidebar.number_input("Trihalomethanes", format="%.2f")
    turbidity = st.sidebar.number_input("Turbidity", format="%.2f")

    # Predict button
    if st.sidebar.button("Predict"):
        input_features = [ph, hardness, solids, sulfate, organic_carbon, chloramines, conductivity, trihalomethanes, turbidity]
        features_value = [np.array(input_features)]

        feature_names = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

        df = pd.DataFrame(features_value, columns=feature_names)
        df = scaler.transform(df)
        output = model.predict(df)

        if output[0] == 1:
            st.markdown('<h2 class="safe">Water is safe for human consumption</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 class="not-safe">Water is not safe for human consumption</h2>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
