import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Solar Forecast", page_icon="🌞", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "LSTM Model",
        "XGBoost Model",
        "Random Forest",
        "Linear Regression",
        "GitHub",
        "About"
    ],
)

# Page Routing
if page == "Home":
    st.title("🏠 Home")
    st.write("Welcome to the Solar Energy Forecasting Dashboard! 🚀")
    st.write("Use the sidebar to navigate between different models and features.")

elif page == "LSTM Model":
    st.title("📊 LSTM Model")
    st.write("This page will show predictions using the LSTM model.")
    epochs = st.slider("Select Number of Epochs", min_value=10, max_value=100, step=10, value=50)
    progress = st.progress(0)
    for i in range(epochs):
        progress.progress((i + 1) / epochs)
    st.success("LSTM Model Training Complete!")

elif page == "XGBoost Model":
    st.title("📈 XGBoost Model")
    st.write("Predictions using XGBoost will be displayed here.")

elif page == "Random Forest":
    st.title("🌲 Random Forest Model")
    st.write("Predictions using the Random Forest model will be displayed here.")

elif page == "Linear Regression":
    st.title("📉 Linear Regression Model")
    st.write("Predictions using Linear Regression will be displayed here.")

elif page == "GitHub":
    st.title("🐙 GitHub Repository")
    st.write("Check out the source code on GitHub.")
    st.markdown("[Visit GitHub](https://github.com/Sivatech24)")

elif page == "About":
    st.title("ℹ️ About")
    st.write("This application forecasts solar energy using different machine learning models.")
    st.write("Developed by Team.")
