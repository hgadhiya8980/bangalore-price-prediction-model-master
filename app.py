from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data//preprocessing_main_house_prediction_file.csv")

X=df.drop("price", axis=1)
y=df["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = joblib.load("house_price_prediction_model.pkl")

def house_price_prediction(model,bath,balcony,total_sqft_int,bhk,price_per_sqft,area_type,availability,location):
    x=np.zeros(len(X.columns))
    x[0]=bath
    x[1]=balcony
    x[2]=total_sqft_int
    x[3]=bhk
    x[4]=price_per_sqft
    
    if "availability"=="Ready To Move":
        x[9]=1
    
    if "area_type_"+area_type in X.columns:
        area_type_index = np.where(X.columns=="area_type_"+area_type)[0][0]
        x[area_type_index]=1
        
    if "location_"+location in X.columns:
        loc_index = np.where(X.columns=="location_"+location)[0][0]
        x[loc_index]=1
        
    x = sc.transform([x])[0]
    
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    bath = request.form["Bathrooms"]
    balcony = request.form["Balcony"]
    total_sqft_int = request.form["Total_Squre_Foot"]
    bhk = request.form["BHK"]
    price_per_sqft = request.form["Price_Per_Squre_Foot"]
    area_type = request.form["Area_Type"]
    availability = request.form["House_Avaliability"]
    location = request.form["House_Location"]
    
    predicated_price =house_price_prediction(model,bath,balcony,total_sqft_int,bhk,price_per_sqft,area_type,availability,location)
    
    return render_template("index.html", prediction_text="Predicated price of bangalore House is {}".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    