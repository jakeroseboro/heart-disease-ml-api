import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report


def predict_heart_disease(data):

    df = pd.read_csv('heart.csv')
    # Remove outliers
    df = df[df.RestingBP >= 84]
    df = df[df.Cholesterol <= 500]

    # Replace string fields with ints
    df['Sex'] = df['Sex'].replace(['F', 'M'], [0,  1])
    df["ChestPainType"] = df["ChestPainType"].replace(["TA", "ATA", "NAP", "ASY"], [0,1,2,3])
    df["RestingECG"] = df["RestingECG"].replace(["Normal", "ST", "LVH"],[0,1,2])
    df["ExerciseAngina"] = df["ExerciseAngina"].replace(["Y", "N"], [0,1])
    df["ST_Slope"] = df["ST_Slope"].replace(["Up", "Flat", "Down"], [0,1,2])

    # Use KNN to replace null values for cholesterol
    df['Cholesterol'].replace(to_replace=0, value=np.nan, inplace=True)
    imputer = KNNImputer(n_neighbors=5)
    fixed = imputer.fit_transform(df)
    cholesterol = []
    for i in range(0, len(df)):
        cholesterol.append(fixed[i][4])
    df["Cholesterol"] = cholesterol

    # Make a copy of the data
    df1 = df.copy()
    df1.drop(columns='HeartDisease', axis=1, inplace=True)

    # Generate test and training data
    x_train, x_test, y_train, y_test = train_test_split(df1, df['HeartDisease'], test_size=0.2, random_state=0)

    clf = RandomForestClassifier(n_estimators=234, min_samples_split=10, min_samples_leaf=2, max_features='sqrt',
                                 max_depth=80, bootstrap=True)

    clf.fit(x_train, y_train)
    y_pred_prob = clf.predict(x_test)
    print(classification_report(y_test, y_pred_prob))

    # predict prob based on user input
    new_x = pd.DataFrame(
        {'Age': [data[0]], 'Sex': [data[1]], 'ChestPainType': [data[2]], 'RestingBP': [data[3]],
         'Cholesterol': [data[4]], 'FastingBS': [data[5]],
         'RestingECG': [data[6]], 'MaxHR': [data[7]], 'ExerciseAngina': [data[8]], 'Oldpeak': [data[9]],
         'ST_Slope': [data[10]]})

    y_pred = clf.predict_proba(new_x)[:,1][0]
    return str(round(y_pred, 4))


# This is used to return the best params for the random forest classifier.
# It only needed to be run once and I used the printed values as the params.
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


def heart_disease_stats():
    df = pd.read_csv('heart.csv')
    df_positive = df[df.HeartDisease != 0]
    df_negative = df[df.HeartDisease != 1]

    df["age_range"]= pd.cut(df["Age"], bins=[0, 20, 40, 60, 80, 100])
    df2 = df.groupby("age_range").HeartDisease.sum().to_frame(name="heart_disease")
    df2["no_heart_disease"] = df.groupby("age_range").HeartDisease.count() - df2.heart_disease
    df2 = df2.reset_index()

    df["resting_bp"] = pd.cut(df["RestingBP"], bins=[-1, 120, 240])
    df2_bp = df.groupby("resting_bp").HeartDisease.sum().to_frame(name="heart_disease")
    df2_bp["no_heart_disease"] = df.groupby("resting_bp").HeartDisease.count() - df2_bp.heart_disease
    df2_bp = df2_bp.reset_index()

    df["fasting_bs"] = pd.cut(df["FastingBS"], bins=[-1, 0,1])
    df2_bs = df.groupby("fasting_bs").HeartDisease.sum().to_frame(name="heart_disease")
    df2_bs["no_heart_disease"] = df.groupby("fasting_bs").HeartDisease.count() - df2_bs.heart_disease
    df2_bs = df2_bs.reset_index()

    df["max_hr"] = pd.cut(df["MaxHR"], bins=[59, 131, 202])
    df2_max_hr = df.groupby("max_hr").HeartDisease.sum().to_frame(name="heart_disease")
    df2_max_hr["no_heart_disease"] = df.groupby("max_hr").HeartDisease.count() - df2_max_hr.heart_disease
    df2_max_hr = df2_max_hr.reset_index()

    df["old_peak"] = pd.cut(df["Oldpeak"], bins=[-7, 1, 7])
    df2_oldpeak = df.groupby("old_peak").HeartDisease.sum().to_frame(name="heart_disease")
    df2_oldpeak["no_heart_disease"] = df.groupby("old_peak").HeartDisease.count() - df2_oldpeak.heart_disease
    df2_oldpeak = df2_oldpeak.reset_index()

    data = {
        "age":{
            "20_40_heart_disease": int(df2.loc[[1]].heart_disease.values[0]),
            "20_40_no_heart_disease": int(df2.loc[[1]].no_heart_disease.values[0]),
            "40_60_heart_disease": int(df2.loc[[2]].heart_disease.values[0]),
            "40_60_no_heart_disease": int(df2.loc[[2]].no_heart_disease.values[0]),
            "60_80_heart_disease": int(df2.loc[[3]].heart_disease.values[0]),
            "60_80_no_heart_disease": int(df2.loc[[3]].no_heart_disease.values[0]),
        },
        "sex":{
            "m_positive": int(df_positive.value_counts(['Sex']).M),
            "m_negative": int(df_negative.value_counts(['Sex']).M),
            "f_positive": int(df_positive.value_counts(['Sex']).F),
            "f_negative": int(df_negative.value_counts(['Sex']).F)
        },
        "chest_pain_type":{
            "ata_positive": int(df_positive.value_counts(['ChestPainType']).ATA),
            "ata_negative": int(df_negative.value_counts(['ChestPainType']).ATA),
            "ta_positive": int(df_positive.value_counts(['ChestPainType']).TA),
            "ta_negative": int(df_negative.value_counts(['ChestPainType']).TA),
            "nap_positive": int(df_positive.value_counts(['ChestPainType']).NAP),
            "nap_negative": int(df_negative.value_counts(['ChestPainType']).NAP),
            "asy_positive": int(df_positive.value_counts(['ChestPainType']).ASY),
            "asy_negative": int(df_negative.value_counts(['ChestPainType']).ASY),
        },
        "resting_bp":{
            "resting_bp_under_120_positive": int(df2_bp.loc[[0]].heart_disease.values[0]),
            "resting_bp_under_120_negative": int(df2_bp.loc[[0]].no_heart_disease.values[0]),
            "resting_bp_over_120_positive": int(df2_bp.loc[[1]].heart_disease.values[0]),
            "resting_bp_over_120_negative": int(df2_bp.loc[[1]].no_heart_disease.values[0]),
        },
        "fasting_bs": {
            "bs_over_120_positive": int(df2_bs.loc[[1]].heart_disease.values[0]),
            "bs_over_120_negative": int(df2_bs.loc[[1]].no_heart_disease.values[0]),
            "bs_under_120_positive": int(df2_bs.loc[[0]].heart_disease.values[0]),
            "bs_under_120_negative": int(df2_bs.loc[[0]].no_heart_disease.values[0])
        },
        "resting_ecg":{
            "normal_positive": int(df_positive.value_counts(['RestingECG']).Normal),
            "normal_negative": int(df_negative.value_counts(['RestingECG']).Normal),
            "ST_positive": int(df_positive.value_counts(['RestingECG']).ST),
            "ST_negative": int(df_negative.value_counts(['RestingECG']).ST),
            "LVH_positive": int(df_positive.value_counts(['RestingECG']).LVH),
            "LVH_negative": int(df_negative.value_counts(['RestingECG']).LVH),
        },
        "max_hr": {
            "60_131_heart_disease": int(df2_max_hr.loc[[0]].heart_disease.values[0]),
            "60_131_no_heart_disease": int(df2_max_hr.loc[[0]].no_heart_disease.values[0]),
            "131_202_heart_disease": int(df2_max_hr.loc[[1]].heart_disease.values[0]),
            "131_202_no_heart_disease": int(df2_max_hr.loc[[1]].no_heart_disease.values[0]),
        },
        "exercise_angina":{
            "exercise_angina_positive": int(df_positive.value_counts(['ExerciseAngina']).Y),
            "exercise_angina_negative": int(df_negative.value_counts(['ExerciseAngina']).Y),
            "no_exercise_angina_positive": int(df_positive.value_counts(['ExerciseAngina']).N),
            "no_exercise_angina_negative": int(df_negative.value_counts(['ExerciseAngina']).N)
        },
        "old_peak":{
            "-7_1_heart_disease": int(df2_oldpeak.loc[[0]].heart_disease.values[0]),
            "-7_1_no_heart_disease": int(df2_oldpeak.loc[[0]].no_heart_disease.values[0]),
            "1_7_heart_disease": int(df2_oldpeak.loc[[1]].heart_disease.values[0]),
            "1_7_no_heart_disease": int(df2_oldpeak.loc[[1]].no_heart_disease.values[0]),
        },
        "st_slope":{
            "Up_positive": int(df_positive.value_counts(['ST_Slope']).Up),
            "Up_negative": int(df_negative.value_counts(['ST_Slope']).Up),
            "Flat_positive": int(df_positive.value_counts(['ST_Slope']).Flat),
            "Flat_negative": int(df_negative.value_counts(['ST_Slope']).Flat),
            "Down_positive": int(df_positive.value_counts(['ST_Slope']).Down),
            "Down_negative": int(df_negative.value_counts(['ST_Slope']).Down),
        }

    }
    return data
