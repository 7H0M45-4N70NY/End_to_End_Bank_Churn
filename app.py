from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from Bank_Churn.pipeline.stage_06_prediction import PredictionPipline,CustomData

app=Flask(__name__)

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="GET":
        return render_template("form.html")
    else:

        data=CustomData(
            Geography=str(request.form.get("Geography")),
            Gender=str(request.form.get("Gender")),
            CreditScore=float(request.form.get("CreditScore")),
            Age=float(request.form.get("Age")),
            Tenure=float(request.form.get("Tenure")),
            Balance=float(request.form.get("Balance")),
            NumOfProducts=float(request.form.get("NumOfProducts")),
            HasCrCard=int(bool(request.form.get("HasCrCard"))),
            IsActiveMember=int(bool(request.form.get("IsActiveMember"))),
            EstimatedSalary=float(request.form.get("EstimatedSalary"))
        )
        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictionPipline()

        pred=predict_pipeline.predict(final_data)

        result=round(pred[0],2)
        res=""
        if result==0:
            res="Not Churn"
        else:
            res="Churn"

        return render_template("result.html",final_result=res)




if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080,debug=True)