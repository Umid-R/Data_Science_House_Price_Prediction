from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crime_rate         = float(request.form['crime_rate'])
    residential_zone   = float(request.form['residential_zone_pct'])
    industrial_pct     = float(request.form['industrial_pct'])
    charles_river      = int(request.form['charles_river'])
    nitric_oxide       = float(request.form['nitric_oxide'])
    avg_rooms          = float(request.form['avg_rooms'])
    pre1940_pct        = float(request.form['pre1940_pct'])
    employment_dist    = float(request.form['employment_dist'])
    highway_access     = float(request.form['highway_access'])
    tax_rate           = float(request.form['tax_rate'])
    pupil_teacher      = float(request.form['pupil_teacher_ratio'])
    black_proportion   = float(request.form['black_proportion'])
    lower_status_pct   = float(request.form['lower_status_pct'])

    input_data = pd.DataFrame({
        'crime_rate':           [crime_rate],
        'residential_zone_pct': [residential_zone],
        'industrial_pct':       [industrial_pct],
        'charles_river':        [charles_river],
        'nitric_oxide':         [nitric_oxide],
        'avg_rooms':            [avg_rooms],
        'pre1940_pct':          [pre1940_pct],
        'employment_dist':      [employment_dist],
        'highway_access':       [highway_access],
        'tax_rate':             [tax_rate],
        'pupil_teacher_ratio':  [pupil_teacher],
        'black_proportion':     [black_proportion],
        'lower_status_pct':     [lower_status_pct],
    })

    log_pred = model.predict(input_data)[0]
    predicted_price = round(float(np.exp(log_pred)), 2)

    return render_template('index.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)