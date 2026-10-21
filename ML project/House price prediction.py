import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data={
       "Area":[800,1000,1200,1500,1800,2000,2200,2500],
       "Bedrooms":[2,2,3,3,4,4,5,5],
       "Bathrooms":[1,2,2,3,3,4,4,5],
       "Age":[20,15,10,8,5,4,3,2],
       "Location":["jaipur","jaipur","delhi","delhi","mumbai","jaipur","mumbai","delhi"],
       "price":[40,50,60,75,90,100,115,125]
}
df=pd.DataFrame(data)
print(df)
print("/n Dataset information:")
print(df.info())
print("/n Missing values:")
print(df.isnull().sum())
print("/n Dataset statistics:")
print(df.describe())
X=df[["Area","Bedrooms","Bathrooms","Age","Location"]]
y=df["price"]
X=pd.get_dummies(X,columns=["Location"],dtype=int)
print("/n After Encoding:")
print(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("/n Actual price: ")
print(y_test.values)
print("/n Predicted price:")
print(predictions)
mae=mean_absolute_error(y_test,predictions)
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
r2=r2_score(y_test,predictions)
print("/n || MODEL EVALUTION ||")
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:",r2)
result=pd.DataFrame({"Actual price":y_test.values, "predicted price": predictions})
print("/n==== ACTUAL VS PREDICTED====")
print(result)
new_house= pd.DataFrame({
    "Area":[1000],
    "Bedrooms":[3],
    "Bathrooms":[1],
    "Age":[4],
    "Location":["Delhi"]
})
new_house=pd.get_dummies(new_house,columns=["Location"],dtype=int)
new_house=new_house.reindex(columns=X.columns,fill_value=0)
prediction=model.predict(new_house)
print(" /n predicted house price:",prediction[0])
joblib.dump(model,"house_price_model.pkl")
print(("/n model saved successfully "))
loaded_model=joblib.load("house_price_model.pkl")
print("Model loaded successfully!")


