# importing dependencies
import numpy as np
import pickle
import streamlit as st
import pandas as pd

# loading the model
model_package = pickle.load(open('diabetes-pred-model.sav', 'rb'))

# extracting model, scaler, and column names
model = model_package["model"]
scaler = model_package["scaler"]
columns = model_package["columns"]

# function for the prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # converting to DataFrame with the same column names as the original data
    input_data_df = pd.DataFrame(input_data_reshaped, columns=columns)

    # standardizing the input data
    std_data = scaler.transform(input_data_df)

    # making the prediction
    prediction = model.predict(std_data)

    # returning the result based on the prediction
    if prediction[0] == 0:
        return "is not diabetic"
    else:
        return "is diabetic"

# main function for the web app
def main():
    # setting up the app title
    st.title("Diabetes Prediction Web App")
    
    # adding a placeholder for an image
    st.image("img.jpg")

    # getting input from the user
    name = st.text_input("Name of the Person", key="name")
    
    with st.expander("Fill in Health Details"):
        pregnancies = st.text_input("Number of Pregnancies", key="pregnancies")
        glucose = st.text_input("Glucose Level", key="glucose")
        blood_pressure = st.text_input("Blood Pressure Level", key="blood_pressure")
        skin_thickness = st.text_input("Skin Thickness", key="skin_thickness")
        insulin = st.text_input("Insulin Level", key="insulin")
        bmi = st.text_input("BMI", key="bmi")
        dpf = st.text_input("Diabetes Pedigree Function", key="dpf")
        age = st.text_input("Age of the Person", key="age")

    # handling the predict button click
    if st.button("Predict"):
        # validating inputs
        try:
            pregnancies = int(pregnancies)
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            dpf = float(dpf)
            age = int(age)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
            return

        # preparing the input data
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

        # making the prediction
        diagnosis = diabetes_prediction(input_data)
        
        # displaying the personalized result
        if name:
            st.success(f"{name} {diagnosis}")
        else:
            st.success(f"The person {diagnosis}")

if __name__ == '__main__':
    main()
