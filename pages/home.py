import streamlit as st

# Home Page
st.title("Welcome to the Diabetes Prediction App")

st.header("What is Diabetes?")
st.write(
    """
    Diabetes is a chronic medical condition in which the body is unable to properly regulate blood sugar (glucose) levels. 
    There are two main types of diabetes:
    
    - **Type 1 Diabetes**: This type occurs when the body does not produce insulin, a hormone that regulates blood sugar levels.
    - **Type 2 Diabetes**: This type occurs when the body does not use insulin properly, or does not produce enough insulin.
    
    Both types can lead to serious health complications if not managed properly.
    """
)

st.write(
    "To predict whether you are at risk of diabetes, go to the [Prediction Page](prediction.py)."
)