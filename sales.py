import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import pickle

with open("sales_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Advertising Sales Prediction")
st.write("Enter advertising budgets to predict sales.")

tv_budget = st.number_input("TV Advertising Budget ($)", min_value=0.0, value=100.0)
radio_budget = st.number_input("Radio Advertising Budget ($)", min_value=0.0, value=100.0)
newspaper_budget = st.number_input("Newspaper Advertising Budget ($)", min_value=0.0, value=100.0)

if st.button("Predict Sales"):
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    predicted_sales = model.predict(input_data)[0]
    st.success(f"Estimated Sales: {predicted_sales:.2f}")