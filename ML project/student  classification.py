import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data={"study_hours":[2,3,4,5,6,7,8,3],
       "Attendence":[60,65,70,75,80,85,90,55],
       "previous_marks":[40,45,50,55,60,65,70,40],
       "result":["Fail","Fail","Fail","Fail","passl","pass","pass","Fail"]
}
df=pd.DataFrame(data)
print(df)
X=df[["study_hours","Attendence","previous_marks"]]
y=df["result"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("Actual prediction:")
print(y_test.values)
print("/n predicted results:")
print(prediction)
accuracy=accuracy_score(y_test,prediction)
print("/n Accuracy:",accuracy)
cm=confusion_matrix(y_test,prediction,labels=["pass","Fail"])
print("/n confusion matrix:")
print(cm)
new_student=[[6,85,70]]
predictions=model.predict(new_student)
print("/n New student prediction:",prediction)






