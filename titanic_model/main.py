import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = pd.read_csv('titanic.csv')
titanic_data.info()
print(titanic_data.isnull().sum())

#Data cleaning and Feature engineering #df-dataframe
def preprocess_data(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    fill_missing_ages(df)

    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Age"] = df["Age"].fillna(df["Age"].median())

    #convert Gender to binary
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}) 
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] 
    df["IsAlone"] = np.where(df["FamilySize"] > 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 80, np.inf], labels=False)

    return df


#Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df['Pclass'].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()   

    df['Age'] = df.apply(lambda row: age_fill_map[row['Pclass']] if pd.isnull(row['Age']) else row['Age'], axis=1)

preprocessed_data = preprocess_data(titanic_data)

#Create Features and Target variable
X = preprocessed_data.drop(columns=['Survived'])
y = preprocessed_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   

#ML Model Training and Evaluation
scaler= MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

#Tuning Hyperparameters -- KNN algorithm
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric" : ["euclidean", "manhattan", "minkowski"],
        "weights" : ["uniform", "distance"]
        }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_knn_model = tune_model(X_train, y_train)

#predictions and evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_knn_model, X_test_scaled, y_test)

print (f'Model Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:\n {matrix}')
print(matrix)

#plot confusion matrix
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt='d', xticklabels=["survived", "not survived"], yticklabels=["not survived", "survived"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Value')
    plt.xlabel('Predicted Label')
    plt.show()

plot_model(matrix)

