'''
Steps:
1. Data Loading
2. Data Preprocessing
3. Splitting data into train and test sets
4. Applying SVM with a Linear Kernel -> Since this is a binary classification problem, we will apply the Support Vector Classifier (SVC) with a linear kernel to separate the classes. (SVC also supports multi-class classification)
5. Training the SVM Model
6. Model Evaluation
7. Predictive System
'''

# importing dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading data
try:
    diabetes_dataset = pd.read_csv('diabetes.csv', header=0)   #HEADER=0 -> Use the first row (index 0) as the header (so we can later use them as column names)
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# processing the data
print(diabetes_dataset.head())   # getting an idea of the data 
print(diabetes_dataset.info()) 
print(diabetes_dataset.isnull().sum())   # checking if any column has any null value, if yes we will to adjust for it.
print(diabetes_dataset.describe())   # statistical measure of the data
print(diabetes_dataset['Outcome'].value_counts())   # checking how many total diabetic and non-diabetic samples in the data (0->non-diabetic, 1->diabetic)
print(diabetes_dataset.groupby('Outcome').mean())   # mean for each column value, grouped by 0 and 1


# seperating data and labels
X = diabetes_dataset.drop(columns=['Outcome'], axis=1)   # 1st column is just indexing, axis = 1 ->  operations along the columns
Y = diabetes_dataset['Outcome']

# standardizing the data
scaler = StandardScaler()   #StandardScaler is used to normalize the features of data by removing the mean and scaling it to unit variance.
scaler.fit(X[1:])   #Computes the mean and standard deviation for each feature in X
standardized_data = scaler.transform(X)   #transforming the data
X = standardized_data


# splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=1)   #By using stratify=Y, both subsets will have the same proportion of each class as the original dataset.

# training the model
model = svm.SVC(kernel='linear')   # SVC is the Support Vector Classification model; 'linear' kernel means the model will try to find a linear hyperplane that separates the data into classes
model.fit(X_train,Y_train)

# evaluating the model
X_train_prediction = model.predict(X_train)   #accuracy on training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy on training data: {training_data_accuracy:.2%}')
X_test_prediction = model.predict(X_test)    #accuracy on test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy on test data: {test_data_accuracy:.2%}')

# predicting system
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Converting to DataFrame with the same column names as the original data
input_data_df = pd.DataFrame(input_data_reshaped, columns=diabetes_dataset.columns[:-1])

# standardizing the input data
std_data = scaler.transform(input_data_df)
print(std_data)

prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
