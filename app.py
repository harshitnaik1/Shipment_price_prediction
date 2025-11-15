import pickle
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("shipment_model.pkl","rb"))
preprocessor = pickle.load(open("preprocessor.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ["POST"])
def predict():
    data = {
        'Artist Reputation': float(request.form['artist']),
        'Height': float(request.form['height']),
        'Width': float(request.form['width']),
        'Weight': float(request.form['weight']),
        'Material': request.form['material'],
        'Price Of Sculpture': float(request.form['price']),
        'Base Shipping Price': float(request.form['base_price']),
        'International': request.form['international'],
        'Express Shipment': request.form['express'],
        'Installation Included': request.form['install'],
        'Transport': request.form['transport'],
        'Fragile': request.form['fragile'],
        'Customer Information': request.form['customer'],
        'Remote Location': request.form['remote'],
        'Month': int(request.form['month']),
        'Year': int(request.form['year'])
    }
    new_data = pd.DataFrame([data])
    processed = preprocessor.transform(new_data)
    pred = model.predict(processed)

    final_result = np.expm1(pred)[0]

    return jsonify({"Predicted Shipment Cost":round(final_result,2)})

if __name__=="__main__":
    app.run(debug=True)