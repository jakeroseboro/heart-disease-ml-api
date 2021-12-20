import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def predict_heart_disease(data):

    df = pd.read_csv('heart.csv')
    df = df.drop("Cholesterol", 1)
    df['Sex'] = df['Sex'].replace(['F', 'M'], [0,  1])
    df["ChestPainType"] = df["ChestPainType"].replace(["TA", "ATA", "NAP", "ASY"], [0,1,2,3])
    df["RestingECG"] = df["RestingECG"].replace(["Normal", "ST", "LVH"],[0,1,2])
    df["ExerciseAngina"] = df["ExerciseAngina"].replace(["Y", "N"], [0,1])
    df["ST_Slope"] = df["ST_Slope"].replace(["Up", "Flat", "Down"], [0,1,2])

    new_x = pd.DataFrame(
        {'Age': [data[0]], 'Sex': [data[1]], 'ChestPainType': [data[2]], 'RestingBP': [data[3]], 'FastingBS': [data[4]],
         'RestingECG': [data[5]], 'MaxHR': [data[6]], 'ExerciseAngina': [data[7]], 'Oldpeak': [data[8]], 'ST_Slope': [data[9]]})

    x = df.drop('HeartDisease', 1)

    # Generate test and training data
    x_train, x_test, y_train, y_test = train_test_split(x, df['HeartDisease'], test_size=0.25, random_state=0)

    clf = RandomForestClassifier()

    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(new_x)[:,1]
    return str(y_pred)
