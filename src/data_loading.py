import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    data = load_breast_cancer()
    X, y = data.data ,data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test