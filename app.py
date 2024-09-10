# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# @st.cache_data
# def load_data():
#     file_path = 'data/diabetes.csv'  # Update this path to your CSV file
#     data = pd.read_csv(file_path)
#     return data

# data = load_data()

# # Rename columns in the dataset to match the user input
# data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# # Split data into features (X) and target (Y)
# X = data.drop("Outcome", axis=1)
# Y = data["Outcome"]

# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Initialize and train the Logistic Regression model
# model = LogisticRegression(max_iter=600)
# model.fit(X_train, Y_train)

# # App title
# st.title("Diabetes Prediction App")

# # Input features from the user
# st.sidebar.header('Input Features')

# def user_input_features():
#     pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
#     glucose = st.sidebar.slider('Glucose', 0, 200, 120)
#     blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
#     skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
#     insulin = st.sidebar.slider('Insulin', 0, 846, 79)
#     bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
#     dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.42, 0.3725)
#     age = st.sidebar.slider('Age', 21, 100, 29)
    
#     data = {
#         'Pregnancies': pregnancies,
#         'Glucose': glucose,
#         'BloodPressure': blood_pressure,
#         'SkinThickness': skin_thickness,
#         'Insulin': insulin,
#         'BMI': bmi,
#         'DiabetesPedigreeFunction': dpf,
#         'Age': age
#     }
    
#     features = pd.DataFrame(data, index=[0])
#     return features

# input_df = user_input_features()

# # Display the user input features
# st.subheader('User Input Features')
# st.write(input_df)

# # Scale the input data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X)
# input_scaled = scaler.transform(input_df)

# # Predict the outcome
# prediction = model.predict(input_scaled)[0]

# # Display the prediction
# if st.button('Predict'):
#     if prediction == 1:
#         st.subheader('The model predicts that the person is likely to have diabetes.')
#     else:
#         st.subheader('The model predicts that the person is unlikely to have diabetes.')



import streamlit as st

st.set_page_config(page_title="Diabetes Prediction App", page_icon=":guardsman:", layout="wide")

st.write("Welcome to the Diabetes Prediction App! Use the sidebar to navigate.")
