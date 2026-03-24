from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data.csv")

X = df[["Age","Gender","Total_Spend","Average_Order_Value","Purchase_Frequency","Last_Purchase_Days","Customer_Rating","Complaint_Raised","Return_Count"]]
Y = df["Churn"]

model = RandomForestClassifier()
model.fit(X,Y)
joblib.dump(model, 'trained.pkl')


app = Flask(__name__)

@app.route('/pankaj', methods=["GET","POST"])
def test():
    if request.method == 'POST':
        age = request.form.get("age")
        gender = request.form.get("gender")
        Total_Spend = request.form.get("Total_Spend")
        Average_Order_Value = request.form.get("Average_Order_Value")
        Purchase_Frequency = request.form.get("Purchase_Frequency")
        Last_Purchase_Days = request.form.get("Last_Purchase_Days")
        Customer_Rating = request.form.get("Customer_Rating")
        Complaint_Raised = request.form.get("Complaint_Raised")
        Return_Count = request.form.get("Return_Count")
        
        resultbinary = model.predict([[age,gender,Total_Spend,Average_Order_Value,Purchase_Frequency,Last_Purchase_Days,Customer_Rating,Complaint_Raised,Return_Count]])
        result = model.predict_proba([[age,gender,Total_Spend,Average_Order_Value,Purchase_Frequency,Last_Purchase_Days,Customer_Rating,Complaint_Raised,Return_Count]])
        probofzero = result[0][0]
        probofone = result[0][1]

        if probofzero > probofone:
            final = probofzero
        else:
            final = probofone
        
        
         
        return render_template('success.html', probability = final,result = resultbinary)
        
        # result is : [[0.09 0.91]]
    
    
    
    
    return render_template('hello.html')

if __name__ == '__main__':
    app.run(debug=True)