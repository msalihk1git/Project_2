#import relevant libraries for flask,html rendering and loading the ML model
from flask import Flask,request,url_for,render_template
import pickle
import pandas as pd
import joblib
app = Flask(__name__)
# model=pickle.load(open("model.pkl","rb"))
model=joblib.load(open("model.pkl","rb"))
# scale=pickle.load(open("scale.pkl","rb"))
scale=joblib.load(open("scale.pkl","rb"))


@app.route("/")
def landingPage():
    return render_template("index.html") 

@app.route("/predict",methods=["POST", "GET"])
def predict():
    SatisfactionLevel = request.form['1']
    LastEvaluation = request.form['2']
    NumberOfProject = request.form['3']
    AverageMonthlyHours = request.form['4']
    TimeSpendInCompany = request.form['5']
    WorkAccident = request.form['6']
    PromotionInLast5Years = request.form['7']
    Department = request.form['8']
    Salary = request.form['9']
    rowDf=pd.DataFrame([pd.Series([SatisfactionLevel,LastEvaluation,NumberOfProject,AverageMonthlyHours,TimeSpendInCompany,WorkAccident,PromotionInLast5Years,Department,Salary])])
    rowDf_new=pd.DataFrame(scale.transform(rowDf))
    
    print(rowDf_new)

#  model prediction 
    prediction= model.predict_proba(rowDf_new)
    print(f"The  Predicted values is :{prediction[0][1]}")

    if prediction[0][1] >= 0.5:
        valPred = round(prediction[0][1],3)
        print(f"The Round val {valPred*100}%")
        return render_template('result.html',pred=f'probability of leaving the firm is {valPred*100}%.')
    else:
        valPred = round(prediction[0][0],3)
        return render_template('result.html',pred=f'Probability of staying the firm  is{valPred*100}%.')
if __name__ == "__main__":
    app.run(debug=True)
    

