# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField, SelectField, FloatField
from wtforms.validators import Email, DataRequired, NumberRange

# login and registration


class LoginForm(FlaskForm):
    username = StringField("Username", id="username_login", validators=[DataRequired()])
    password = PasswordField("Password", id="pwd_login", validators=[DataRequired()])


class CreateAccountForm(FlaskForm):
    username = StringField(
        "Username", id="username_create", validators=[DataRequired()]
    )
    email = StringField(
        "Email", id="email_create", validators=[DataRequired(), Email()]
    )
    password = PasswordField("Password", id="pwd_create", validators=[DataRequired()])
    height = FloatField(
        "Height (ft)", validators=[DataRequired(), NumberRange(min=0)]
    )
    weight = FloatField(
        "Weight (kg)", validators=[DataRequired(), NumberRange(min=0)]
    )
    age = IntegerField("Age", validators=[DataRequired(), NumberRange(min=0)])
    gender = SelectField(
        "Gender",
        choices=[("male", "Male"), ("female", "Female")],
        validators=[DataRequired()],
    )
