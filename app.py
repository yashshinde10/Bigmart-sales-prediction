import pandas as pd
from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['Item_Weight'])
    item_fat_content=float(request.form['Item_Fat_Content'])
    item_visibility= float(request.form['Item_Visibility'])
    item_type_combined= float(request.form['Item_Type_Combined'])
    item_mrp = float(request.form['Item_Mrp'])
    outlet_years= float(request.form['Outlet_Years'])
    outlet_size= float(request.form['Outlet_Size'])
    outlet_location_type= float(request.form['Outlet_Location_Type'])
    outlet_type= float(request.form['Outlet_Type'])
    outlet_identifier = float(request.form['Outlet_Identifier'])

    X = np.array([[item_mrp,item_weight, outlet_identifier,outlet_type,outlet_location_type,outlet_size,outlet_years,
        item_type_combined,item_fat_content, item_visibility]])


    model_path = r'C:\Users\PC\Downloads\Store_Sales_Prediction\model\gbr_grid.sav'
    model = joblib.load(model_path)

    Y_pred=model.predict(X)

    return jsonify({'Prediction': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
