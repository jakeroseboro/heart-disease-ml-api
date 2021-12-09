import pandas as pd


def predict_heart_disease(data):
    df = pd.read_csv('heart.csv')
    df = df.drop("Cholesterol", 1)
    print(df)