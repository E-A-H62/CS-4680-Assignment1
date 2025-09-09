# Assignment 1
# References: https://www.kaggle.com/code/imakash3011/water-quality-prediction-7-model/notebook
# See README for analysis, comparison, and evaluation of models

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# import models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# import preprocessing
from sklearn.preprocessing import StandardScaler
# import metrics
from sklearn.metrics import accuracy_score, classification_report

# ignore warnings
warnings.filterwarnings('ignore')
# read data
df = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/1292407/2157486/water_potability.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250908%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250908T032711Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7734b1c4e9e4936b28efa480a33140143112bceb47319184a5cde80240225138841c3dd85a2fddfb33afe665bd1ac9526f26f93e159e1ecdc25cd59b166171e8b0aec67b607583b4c250095344d7487d6f65f817d509a289b0be8d1f40d1a805d7cfcaa99fe9fb5fcb32d8ed063376ae36f27261c82b2861a8ba19c70a5b47fefaa94b0f2ad9a6cb61265ef7fac349d2144ea8d8ae80f0e0fbfbec281b8cfc0b63fdf791de768528f7665a59d4121319d2e56071d90767cb122c8965e96bf6146f9cac4e543be43923e0d24d61f7817f63e95f2d4e840a2d39714faecd874e91382935fbd6cd48e0e4a4a465f7c7260d1c07f426e522e51cb10504cb6cf318e3")
print(df.head())

# print data shape, columns, and dtypes
print("Data Shape:")
print(df.shape)
print(df.columns)
print(df.dtypes)
# print data info and describe
print("Data Info:")
print(df.info())
print(df.describe())
print("Null Values:") # print null values
print(df.isnull().sum())

# drop potability column and create X and y
X = df.drop('Potability', axis=1)
y = df['Potability']

# fill na with mean
X['ph'] = X['ph'].fillna(X['ph'].mean())
X['Sulfate'] = X['Sulfate'].fillna(X['Sulfate'].mean())
X['Trihalomethanes'] = X['Trihalomethanes'].fillna(X['Trihalomethanes'].mean())

# print information about X and y
print("Null Values:")
print(X.isnull().sum())
print("X and y head:") # print head of X and y
print(X.head())
print(y.head())
print("X and y shape:") # print shape of X and y
print(X.shape)
print(y.shape)
print("X and y dtypes:") # print dtypes of X and y
print(X.dtypes)
print(y.dtypes)


# scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print X_train, X_test, y_train, y_test shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# logistic regression model
model1 = LogisticRegression() # initialize model
model1.fit(X_train, y_train) # fit model
y_pred1 = model1.predict(X_test) # predict
print("\nLogistic Regression")
print(y_pred1)
# print accuracy and classification report
print("Accuracy: ", accuracy_score(y_test, y_pred1))
print("Classification Report: ", classification_report(y_test, y_pred1))

# decision tree model
model2 = DecisionTreeClassifier() # initialize model
model2.fit(X_train, y_train) # fit model
y_pred2 = model2.predict(X_test) # predict
print("\nDecision Tree")
print(y_pred2)
# print accuracy and classification report
print("Accuracy: ", accuracy_score(y_test, y_pred2))
print("Classification Report: ", classification_report(y_test, y_pred2))

# support vector machine model  
model3 = SVC() # initialize model
model3.fit(X_train, y_train) # fit model
y_pred3 = model3.predict(X_test) # predict
print("\nSupport Vector Machine")
print(y_pred3)
# print accuracy and classification report
print("Accuracy: ", accuracy_score(y_test, y_pred3))
print("Classification Report: ", classification_report(y_test, y_pred3))

