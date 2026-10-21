import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
data={"study_hours": [2,3,4,5,6,7,8,9],
      "Attendence":[60,65,70,75,80,85,90,95],
      "previous_marks":[40,45,50,55,60,65,70,75,],
      "Marks":[42,47,53,58,63,67,69,74]
}
df=pd.DataFrame(data)
print(df)
X=df[["study_hours","Attendence","previous_marks"]]
y=df["Marks"]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Actual Marks:")
print(y_test.values)
print("predicted Marks:")
print(predictions)
mae=mean_absolute_error(y_test,predictions)
r2=r2_score(y_test,predictions)
print("MAE:",mae)
print("r2 score:",r2)
new_student=[[7,88,72]]
prediction=model.predict((new_student))
print(" \n New student predictions Marks:",prediction)