from flask import Flask, request
from prediction import predict_heart_disease
app = Flask(__name__)


@app.route("/prediction", methods=['POST'])
def prediction_controller():
    json = request.get_json()
    age = json.get("Age")
    sex = json.get("Sex")
    cpt = json.get("ChestPainType")
    rbp = json.get("RestingBP")
    fbs = json.get("FastingBS")
    recg = json.get("RestingECG")
    mhr = json.get("MaxHR")
    ea = json.get("ExerciseAngina")
    op = json.get("Oldpeak")
    sts = json.get("ST_Slope")
    results = [age, sex, cpt, rbp, fbs, recg, mhr, ea, op, sts]
    return predict_heart_disease(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
