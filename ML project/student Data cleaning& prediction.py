import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data={"study_hours":[2,4,None,5,6,7,7],
      "Attendence":[60,70,75,None,85,90,95],
      "subject":["python","ML","ML","python","python","ML","ML"],
      "previous_marks":[40,55,60,70,75,None,85],
      "Result":["Fail","Fail","Fail","pass","pass","pass","pass"]
}
df=pd.DataFrame(data)
print(df)
print("/n missing values:")
print(df.isnull().sum())
df["study_hours"]=df["study_hours"].fillna(0)
df["Attendence"]=df["Attendence"].fillna(0)
df["previous_marks"]=df["previous_marks"].fillna(0)
df=df.drop_duplicates()
print("/n cleaned Dataset:")
print(df)
X=df[["study_hours","Attendence","subject","previous_marks"]]
y=df["Result"]
X=pd.get_dummies(X,columns=["subject"],dtype=int)
print("/n After Encoding:")
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
predictions=model.predict(X_test)
print("/n Actual Results:")
print(y_test.values)
print("/n predicted Result:")
print(predictions)
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)