from flask import Flask, request, jsonify, make_response, redirect, flash, url_for, render_template
from prediction import predict_heart_disease, heart_disease_stats
from flask_cors import CORS
import uuid
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import *
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired, Length, Email
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = str(uuid.uuid4())
cors = CORS(app, resources={r"/*": {"origins": "*"}})
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
Bootstrap(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(String(40), nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    password = Column(String(80), nullable=False)


# WTForm for Login
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=80)])


# WTForm for Sign Up
class SignUpForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired(), Length(min=1, max=40)])
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email')])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=80)])


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/prediction", methods=['POST'])
@login_required
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
@login_required
def heart_disease_data_controller():
    try:
        request_format = request.args.get('format')
        results = heart_disease_stats(request_format)
        return make_response(jsonify(results), 200)
    except:
        return make_response('there was an error', 500)


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    # Initialize form
    form = SignUpForm()

    if form.validate_on_submit():
        # Password security
        hashed_password = generate_password_hash(form.password.data, method="sha256")

        # Check if user already exists in the database
        user_already_exists = User.query.filter_by(email=form.email.data).first()
        if user_already_exists:
            flash("You already have an account with that email address!")
            return redirect(url_for("login"))

        # Create a new user, then go to Login page
        else:
            new_user = User(name=form.name.data, email=form.email.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for("login"))

    return render_template("signup.html", form=form)


# Login page
@app.route("/login", methods=['POST', 'GET'])
def login():
    # Initialize form
    form = LoginForm()

    if form.validate_on_submit():

        # Check to see if email exists in the database
        email_exists = User.query.filter_by(email=form.email.data).first()
        if email_exists:

            # Check to see if the password matches for that email
            if check_password_hash(email_exists.password, form.password.data):
                login_user(email_exists)
                flash("Login successful!")
                next_url = 'http://' + request.form.get("next")
                print(next_url)
                return redirect(next_url)
            else:
                flash("Incorrect password")
                redirect(url_for("login"))
        else:
            flash("You do not have account yet. Please sign up.")
            return redirect(url_for("signup"))

    return render_template('login.html', form=form)


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
