import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data= {
    "Age":[22,35,45,28,52,40,30,60],
    "Monthly_charges":[30,70,90,40,100,80,50,110],
    "Tenure":[2,12,24,5,36,18,8,48],
    "Contract":["Monthly","yearly","yearly","Monthly","yearly","Monthly","Monthly","yearly"],
    "churn":["yes","No","No","yes","yes","No","yes","yes"]
}
df=pd.DataFrame(data)
print(df)
X=df[["Age","Monthly_charges","Tenure","Contract"]]
y=df["churn"]
X=pd.get_dummies(X,columns=["Contract"],dtype=int)
print("/n After Encoding:")
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("/n Actual churn:")
print(y_test.values)
print("/n predicted churn:")
print(predictions)
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
new_customer=[[25,45,4,1,0]]
prediction=model.predict(new_customer)
print("/n New customer:")
print(prediction)
