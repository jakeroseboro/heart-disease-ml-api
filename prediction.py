import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    x_train, x_test, y_train, y_test = train_test_split(x, df['HeartDisease'], test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=234, min_samples_split=10, min_samples_leaf=2, max_features='sqrt',
                                 max_depth=80, bootstrap=True)

    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(new_x)[:,1][0]
    return str(round(y_pred, 4))


def get_best_params(x_train, y_train):

    n_estimators = [int(x) for x in np.linspace(start=103, stop=300, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    print(rf_random.best_params_)