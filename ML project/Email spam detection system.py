import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
data={
    "message":["win a free iphone now","congratulations you won a lottery","get free money now",
                "can we meet tommorrow","are you coming to college","click here to win"," see you at the gym",
                "let's go for lunch","submit your assignment","claim your free prize"],
    "Label":["spam","spam","spam","not spam","not spam","spam","not spam","not spam","not spam","spam"]
}
df=pd.DataFrame(data)
print(df)
X=df["message"]
y=df["Label"]
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
print("/n Training samples:",X_train.shape[0])
print("/n Testing samples:",X_test.shape[0])
model=LogisticRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("/n Actual :")
print(y_test.values)
print("/n Predicted: ")
print(predictions)
accuracy=accuracy_score(y_test,predictions)
print(accuracy)
cm=confusion_matrix(y_test,predictions,labels=["spam","not spam"])
print(cm)
new_message=["congratulations you win a lottery"]
new_message_vector=vectorizer.transform(new_message)
new_prediction=model.predict(new_message_vector)
print("/n====New Mesaage====")
print("message:",new_message[0])
print("prediction:",new_prediction[0])