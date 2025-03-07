import streamlit as st

# Set Streamlit page configuration
st.set_page_config(page_title="Solar Forecast", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "LSTM Model", "XGBoost Model", "Random Forest", "Linear Regression", "GitHub", "About"])

# Redirect to the selected page
if page == "Home":
    from pages import home
    home.show()
elif page == "LSTM Model":
    from pages import lstm
    lstm.show()
elif page == "XGBoost Model":
    from pages import xgboost
    xgboost.show()
elif page == "Random Forest":
    from pages import random_forest
    random_forest.show()
elif page == "Linear Regression":
    from pages import linear_regression
    linear_regression.show()
elif page == "GitHub":
    from pages import github
    github.show()
elif page == "About":
    from pages import about
    about.show()
