from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)
linear_model=pickle.load(open('LinearModel_Regr.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    models=sorted(car['name'].unique())
    years=sorted(car['year'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())
    return render_template('index.html',companies=companies,models=models,years=years,fuel_type=fuel_type)


@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('commpany')
    car_model=request.form.get('model')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    kilo_driven=request.form.get('kdriven')

    prediction=linear_model.predict(pd.DataFrame([[company,car_model,year,fuel_type,kilo_driven]],columns=['name','company','year','fuel_type','kms_driven']))

    return str(prediction[0])

if __name__=='__main__':
    app.run(debug=True)
