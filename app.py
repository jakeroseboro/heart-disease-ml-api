from flask import Flask, request, jsonify, make_response
from prediction import predict_heart_disease
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


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
    data = [age, sex, cpt, rbp, fbs, recg, mhr, ea, op, sts]
    results = predict_heart_disease(data)
    return make_response(jsonify(results), 200)


if __name__ == '__main__':
    app.run()
