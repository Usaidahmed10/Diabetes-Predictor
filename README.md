
# Diabetes Prediction System

This project is a **Diabetes Prediction System** built using **Python**, **Streamlit**, and **Scikit-learn**. It leverages a **Support Vector Machine (SVM)** model to predict whether a person is diabetic based on various health metrics.

---

## Features

- Data preprocessing with scaling using **StandardScaler**
- Model training with **SVM (Linear Kernel)** for binary classification
- **Streamlit** web app for user interaction
- Allows users to input health details and get predictions
- Displays personalized results with the entered name

---

## Steps to Run the Project

### 1. Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- Required Python packages (install via the `requirements.txt`)

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3. Train the Model
1. Run the `predictor.py` file to preprocess the data and train the SVM model:
   ```bash
   python predictor.py
   ```
2. This will generate a file named `diabetes-pred-model.sav` containing the trained model.

### 4. Run the Web App
1. Open the `web_app.py` file to launch the Streamlit app:
   ```bash
   streamlit run web_app.py
   ```
2. Enter health details into the provided fields and click "Predict" to get the results.

---

## Input Data Fields

The web app accepts the following inputs for prediction:
- **Name**: Name of the person
- **Number of Pregnancies**
- **Glucose Level**
- **Blood Pressure Level**
- **Skin Thickness**
- **Insulin Level**
- **BMI** (Body Mass Index)
- **Diabetes Pedigree Function**
- **Age**

---

## Project Structure

```
.
├── predictor.py          # Script to train the model
├── web_app.py            # Streamlit-based web app
├── diabetes.csv          # Dataset for training (add your own)
├── diabetes-pred-model.sav  # Saved model (generated after training)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
```

---

## Dataset

The dataset used for this project is the **PIMA Indian Diabetes Dataset**. Ensure the dataset file `diabetes.csv` is in the same directory as the `predictor.py` file before running the training script.

---

## Requirements

The project dependencies are listed below:
- pandas
- numpy
- scikit-learn
- streamlit
- pickle

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Example Usage

### Prediction Example:
Input data:
```
Pregnancies: 5
Glucose: 166
Blood Pressure: 72
Skin Thickness: 19
Insulin: 175
BMI: 25.8
Diabetes Pedigree Function: 0.587
Age: 51
```

Output:
```
The person is diabetic
```

---

## Screenshots

1. **Web App Interface**  
  ![image](https://github.com/user-attachments/assets/0ca6ebb1-f841-4a06-a229-51f63ccf99a8)


## Acknowledgments

- **Scikit-learn**: For providing a powerful library for machine learning
- **Streamlit**: For creating an interactive and easy-to-use web app
- **PIMA Indian Diabetes Dataset**: For the dataset used in this project

---

