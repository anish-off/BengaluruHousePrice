import pandas as pd
import numpy as np
import pickle
from flask import Flask , render_template , request

app = Flask(__name__)
data = pd.read_csv('Bengaluru_House_Data_Cleaned.csv')
pipe = pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    print(locations)
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    total_sqft = request.form.get('total_sqft')

    print(location,bhk,bath,total_sqft)
    input = pd.DataFrame([[location,bhk,bath,total_sqft]],columns=['location','bhk','bath','total_sqft'])
    prediction = pipe.predict(input)[0]* 1e5
    
    return str(np.round(prediction,2))

if __name__=='__main__':
    app.run(debug=True,port=5000)
