import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data={
    "Age":[22,25,28,30,32,35,38,40,45,50],
    "Experience":[0,2,5,8,10,11,13,15,20,25],
    "Education":["Bachelor","Bachelor","Master","Master","Master","Master","Master","phd","phd","phd"],
    "job_Role":["junior","junior","junior","Developer","Developer","Developer","senior","senior","Manager","Manager"],
    "Salary":[25000,30000,40000,50000,55000,60000,75000,90000,105000,130000]
}
df=pd.DataFrame(data)
print(df)
print("/n===Missing values===")
print(df.isnull().sum())
X=df[["Age","Experience","Education","job_Role"]]
y=df["Salary"]
X=pd.get_dummies(X,columns=["Education","job_Role"],dtype=int)
print("/n====AFTER ENCODING=====")
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("/n Training samples:",len(X_train))
print("/n Testing samples:",len(X_test))
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("/n Actual Salary:")
print(y_test.values)
print("/n predicted Salary:")
print(predictions)
mae=mean_absolute_error(y_test,predictions)
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
r2=r2_score(y_test,predictions)
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:",r2)
results=pd.DataFrame({"Actual salary":y_test.values,"predicted salary":predictions})
print("/n==== ACTUAL VS PREDICTED ====")
print(results)
new_employee=pd.DataFrame({
    "Age":[29],
    "Experience":[5],
    "Education":["Master"],
    "job_Role":["Developer"]
})
new_employee=pd.get_dummies(new_employee,columns=["Education","job_Role"],dtype=int)
new_employee=new_employee.reindex(columns=X.columns,fill_value=0)
prediction=model.predict(new_employee)
print(("n====New Employeee prediction====="))
print("predicted Salary:",prediction[0])
joblib.dump(model,"employee_salary_model.pkl")
print("/n Model saved successfully!")
loaded_model=joblib.load("employee_salary_model.pkl")
print("Model loaded successfuly!")