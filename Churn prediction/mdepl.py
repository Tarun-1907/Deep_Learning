# import sklearn
# print(sklearn.__version__)
import pickle
import pandas as pd
from sklearn import metrics
from flask import Flask, request, render_template

app=Flask("__name__")
df_l=pd.read_csv("Customer-Churn.csv")
q=""

@app.route("/")
def loadpage():
    return render_template('home.html',query='')


@app.route("/",methods=['POST'])
def predict():
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneServices
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    
    input1=request.form['q1']
    input2=request.form['q2']
    input3=request.form['q3']
    input4=request.form['q4']
    input5=request.form['q5']
    input6=request.form['q6']
    input7=request.form['q7']
    input8=request.form['q8']
    input9=request.form['q9']
    input10=request.form['q10']
    input11=request.form['q11']
    input12=request.form['q12']
    input13=request.form['q13']
    input14=request.form['q14']
    input15=request.form['q15']
    input16=request.form['q16']
    input17=request.form['q17']
    input18=request.form['q18']
    input19=request.form['q19']
    
    
    model=pickle.load(open("model.sav",'rb'))
    
    
    data=[[input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12,input13,input14,input15,input16,input17,input18,input19]]
    
    new_df=pd.DataFrame(data,columns=[['SeniorCitizen', 'MonthlyCharges','TotalCharges','gender','Partner','Dependents','PhoneServices','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','tenure']])
    
    df_2=pd.concat([df_l,new_df],ignore_index=True)
    
    labels=["{0}-{1}".format(i,i+11) for i in range(1,72,12)]
    df_2['tenure_group']=pd.cut(df_2.tenure.astype(int),range(1,88,12),right=False,labels=labels)
    df_2.drop(columns=['tenure'],axis=1,inplace=True)

    new_df_dummies=pd.get_dummies(df_2[['gender','SeniorCitizen','Partner','Dependents','PhoneServices','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','tenure_group']])
    
    single=model.predict(new_df_dummies.tail(1))
    probability=model.predict_proba(new_df_dummies.tail(1))[:,1]
    
    if single==1:
        o1="This customer is likely to be churned!!"
        o2="Confidence: {}".format(probability*100)
    else:
        o1="This customer is likely to continue!!"
        o2="Confidence: {}".format(probability*100)
        
    return render_template("home.html",output1=o1,output2=o2,
    q1=request.form['query1'],
    q2=request.form['q2'],
    q3=request.form['q3'],
    q4=request.form['q4'],
    q5=request.form['q5'],
    q6=request.form['q6'],
    q7=request.form['q7'],
    q8=request.form['q8'],
    q9=request.form['q9'],
    q10=request.form['q10'],
    q11=request.form['q11'],
    q12=request.form['q12'],
    q13=request.form['q13'],
    q14=request.form['q14'],
    q15=request.form['q15'],
    q16=request.form['q16'],
    q17=request.form['q17'],
    q18=request.form['q18'],
    q19=request.form['q19'])
    
if __name__=="__main__":
    app.run(port=3000,  debug=True)