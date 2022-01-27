from flask import Flask, request, jsonify, make_response
from prediction import predict_heart_disease, heart_disease_stats
import flask_cors
import uuid
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *
import flask_praetorian

db = SQLAlchemy()
guard = flask_praetorian.Praetorian()
cors = flask_cors.CORS()


class User(db.Model):
    id = Column(Integer, primary_key=True)
    username = Column(Text, unique=True)
    hashed_password = Column(Text)
    roles = Column(Text)
    is_active = Column(Boolean, default=True, server_default="true")

    @property
    def identity(self):
        """
        *Required Attribute or Property*

        flask-praetorian requires that the user class has an ``identity`` instance
        attribute or property that provides the unique id of the user instance
        """
        return self.id

    @property
    def rolenames(self):
        """
        *Required Attribute or Property*

        flask-praetorian requires that the user class has a ``rolenames`` instance
        attribute or property that provides a list of strings that describe the roles
        attached to the user instance
        """
        try:
            return self.roles.split(",")
        except Exception:
            return []

    @property
    def password(self):
        """
        *Required Attribute or Property*

        flask-praetorian requires that the user class has a ``password`` instance
        attribute or property that provides the hashed password assigned to the user
        instance
        """
        return self.hashed_password

    @classmethod
    def lookup(cls, username):
        """
        *Required Method*

        flask-praetorian requires that the user class implements a ``lookup()``
        class method that takes a single ``username`` argument and returns a user
        instance if there is one that matches or ``None`` if there is not.
        """
        return cls.query.filter_by(username=username).one_or_none()

    @classmethod
    def identify(cls, id):
        """
        *Required Method*

        flask-praetorian requires that the user class implements an ``identify()``
        class method that takes a single ``id`` argument and returns user instance if
        there is one that matches or ``None`` if there is not.
        """
        return cls.query.get(id)

    def is_valid(self):
        return self.is_active


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['JWT_ACCESS_LIFESPAN'] = {'hours': 24}
app.config['JWT_REFRESH_LIFESPAN'] = {'days': 30}
app.config['SECRET_KEY'] = str(uuid.uuid4())
db.init_app(app)
guard.init_app(app, User)
cors.init_app(app)


@app.route("/prediction", methods=['POST'])
@flask_praetorian.auth_required
def prediction_controller():
    try:
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
        chl = json.get("Cholesterol")
        data = [age, sex, cpt, rbp, chl, fbs, recg, mhr, ea, op, sts]
        print(data)
        results = predict_heart_disease(data)
        return make_response(jsonify(results), 200)
    except:
        return make_response('There was an error', 500)


@app.route("/data", methods=['GET'])
@flask_praetorian.auth_required
def heart_disease_data_controller():
    try:
        request_format = request.args.get('format')
        results = heart_disease_stats(request_format)
        return make_response(jsonify(results), 200)
    except:
        return make_response('There was an error', 500)


# Login
@app.route("/login", methods=['POST'])
def login():
    try:
        req = request.get_json(force=True)
        username = req.get('username', None)
        password = req.get('password', None)
        user = guard.authenticate(username, password)
        return make_response({"auth_token": guard.encode_jwt_token(user)}, 200)
    except:
        return make_response('There was an error', 500)


@app.route("/signup", methods =['POST'])
def signup():
    try:
        req = request.get_json(force=True)
        username = req.get('username', None)
        password = req.get('password', None)
        if not (User.lookup(username=username)):
            db.session.add(
                User(
                    username=username,
                    hashed_password=guard.hash_password(password),
                    roles="operator",
                )
            )
            db.session.commit()
            user = guard.authenticate(username, password)
            return make_response({"auth_token": guard.encode_jwt_token(user)}, 200)
        else:
            return make_response("User already exists. Please log in.", 500)
    except:
        return make_response('There was an error', 500)


@app.route("/refresh", methods=['POST'])
def refresh():
    try:
        old_token = request.get_data()
        new_token = guard.refresh_jwt_token(old_token)
        return make_response(new_token)
    except:
        return make_response('There was an error', 500)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
