# heart-disease-ml-api
This is a machine learning project built in Python/Flask. It uses SkLearn's random forest classifier to predict the probability of heart disease based on medical history input. The API is hosted on Heroku.
I removed cholesterol from the calculation do to too many empty entries for the feild. 

### Source
The referenced dataset comes from: https://www.kaggle.com/fedesoriano/heart-failure-prediction

### How to use
Send a POST request to the following link: https://heart-disease-ml-api.herokuapp.com/prediction
The body of your request must the following data in JSON format: 
```
{
    "Age": 60,
    "Sex": M,
    "ChestPainType":TA,
    "RestingBP": 190,
    "FastingBS": 1,
    "RestingECG": LVH,
    "MaxHR": 100,
    "ExerciseAngina":Y,
    "Oldpeak":2.0,
    "ST_Slope":Up
}
```

- Age: age of the patient [years]
- Sex: sex of the patient [F: Female, M: Male]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure (systolic) [mm Hg]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
