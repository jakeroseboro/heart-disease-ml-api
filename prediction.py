import pandas as pd


def predict_heart_disease(data):
    df = pd.read_csv('heart.csv')
    df = df.drop("Cholesterol", 1)
    df['Sex'] = df['Sex'].replace(['F', 'M'], [0,  1])
    df["ChestPainType"] = df["ChestPainType"].replace(["TA", "ATA", "NAP", "ASY"], [0,1,2,3])
    df["RestingECG"] = df["RestingECG"].replace(["Normal", "ST", "LVH"],[0,1,2])
    df["ExerciseAngina"] = df["ExerciseAngina"].replace(["Y", "N"], [0,1])
    df["ST_Slope"] = df["ST_Slope"].replace(["Up", "Flat", "Down"], [0,1,2])

    print(df)
    return "HEY"
