from flask import Flask, render_template, request
from ast import literal_eval
from models import *
import json
from pathlib import Path
import os



app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

current_dir = Path().absolute()
naive_model_path = os.path.join(current_dir, 'models\\naive_model.joblib')
decision_model_path = os.path.join(current_dir, 'models\\decision_model.joblib')

@app.route('/')
def my_app():
    return render_template("index.html")