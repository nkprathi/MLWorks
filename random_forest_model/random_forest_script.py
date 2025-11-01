#Implementation of Random Forest in Classification
#Problem statement: To predic whether a bank currency note is authentic or not based on 4 attributes(Variance of thje image, Wavelet transformed image, skewness entropy and kurtosis of the image).
import pandas as pd 
import numpy as np

#import the dataset
dataset = pd.read_csv("data_banknote_authentication.csv")
print(dataset.head())

#Separate the attributes and labels 
X= dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

#Preprocessing the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() #calling the StandardScaler class
X_train = sc.fit_transform(X_train) #fit and transform the training data
X_test = sc.transform(X_test) #transform the test data

#Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier #ensemble module contains the RandomForestClassifier class
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#parameter defines the number of trees in the Randomforest
#For classsification problem, the metrics used is accuracy confusion matrix, precision, recall and f1-score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
