from flask import Flask
from flask import request
from flask import render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pickle.load(open('models/df.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    # Sort the unique values
    companies = sorted(df['Company'].unique())
    types = sorted(df['TypeName'].unique())
    cpus = sorted(df['Cpu brand'].unique())
    gpus = sorted(df['Gpu brand'].unique())
    os_list = sorted(df['os'].unique())

    return render_template('index.html',
                           companies=companies,
                           types=types,
                           cpus=cpus,
                           gpus=gpus,
                           os_list=os_list)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    company = request.form['company']
    type = request.form['type']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    ips = 1 if request.form['ips'] == 'Yes' else 0
    screen_size = float(request.form['screen_size'])
    resolution = request.form['resolution']
    cpu = request.form['cpu']
    hdd = int(request.form['hdd'])
    ssd = int(request.form['ssd'])
    gpu = request.form['gpu']
    os = request.form['os']

    # Calculate PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Prepare query
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Sort the unique values again for consistent dropdown options
    companies = sorted(df['Company'].unique())
    types = sorted(df['TypeName'].unique())
    cpus = sorted(df['Cpu brand'].unique())
    gpus = sorted(df['Gpu brand'].unique())
    os_list = sorted(df['os'].unique())

    # Render template with prediction and input values
    return render_template(
        'index.html',
        companies=companies,
        types=types,
        cpus=cpus,
        gpus=gpus,
        os_list=os_list,
        price=predicted_price,
        company=company,
        type=type,
        ram=ram,
        weight=weight,
        touchscreen=touchscreen,
        ips=ips,
        screen_size=screen_size,
        resolution=resolution,
        cpu=cpu,
        hdd=hdd,
        ssd=ssd,
        gpu=gpu,
        os=os
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)