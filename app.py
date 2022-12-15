from flask import Flask, render_template, request, jsonify
import csv
import io
import joblib
import pandas as pd
from keras.models import load_model

dtree = joblib.load('models/dt.joblib')
nbayes = joblib.load('models/nb.joblib')
ann = joblib.load('models/ann.joblib')

feature_labels = {
    'parent': 'Parent',
    'has_nurs': 'Has Nursery',
    'form': 'Form',
    'children': 'Children',
    'housing': 'Housing',
    'finance': 'Finance',
    'social': 'Social',
    'health': 'Health',
}

outcomes = ['not_recom', 'priority', 'spec_prior', 'very_recom']

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.debug = True


@app.route('/')
def index(name=None):
    return render_template("index.html")


@app.post('/result')
def result():

    inputs = []
    prediction = '?'

    data = request.form
    data = request.form
    parent = request.form['parent']
    has_nurs = request.form['has_nurs']
    form = request.form['form']
    children = request.form['children']
    housing = request.form['housing']
    finance = request.form['finance']
    social = request.form['social']
    health = request.form['health']
    inputs = [
        parent, has_nurs, form, children, housing, finance, social, health
    ]
    df_inputs = pd.DataFrame([inputs], columns=feature_labels.keys())
    classifier = request.form.get('classifier')
    if classifier == 'Decision Tree':
        prediction = outcomes[dtree.predict(df_inputs)[0]]
    elif classifier == 'Naive Bayes':
        prediction = outcomes[nbayes.predict(df_inputs)[0]]
    elif classifier == 'ANN':
        prediction = outcomes[ann.predict(df_inputs)[0]]
    else:
        prediction = 'invalid classifier'

    # test_input = request.files.get('test_input_file')
    # test_input.stream.seek(0)
    # test_stream = io.StringIO(test_input.stream.read().decode('UTF8'))
    # csv_reader = csv.DictReader(test_stream, delimiter=',', quotechar='"')
    # for row in csv_reader:
    #     parent = row['parent']
    #     has_nurs = row['has_nurs']
    #     form = row['form']
    #     children = row['children']
    #     housing = row['housing']
    #     finance = row['finance']
    #     social = row['social']
    #     health = row['health']
    return prediction


if __name__ == "__main__":
    app.run(debug=True)