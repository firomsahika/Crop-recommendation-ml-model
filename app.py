from flask import Flask, request, render_template

import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods =['POST'])
def predict():
    N = request.form['Nitrogen']
    K = request.form['Potassium']
    P = request.form['Phosporus']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']


    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1,-1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {
        1:"Rice",2:"Maize",3:"Chickpea", 4:"Kidneybeans",5:"pigeonpeas",
        6: "Mothbeans",7:"Mungbean",8:"Blackgram",9:"Lentil", 10:"Pomegranate",
        11:"Banana",12:"Mango",13:"Grapes",14:"Watermelon",15:"Muskmelon",16:"Apple",
        17:"Orange",18:"Papaya",19:"Coconut",20:"Cotton",21:"Jute",22:"Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)

    else:
        result = "Sorry, we coudn't determine the best crop to be cultivated right there!"

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)