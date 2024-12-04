import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
iris=load_iris()
feature=iris.data
label=iris.target
df=pd.DataFrame(feature,columns=iris.feature_names)
print(df.head())
print((pd.DataFrame(label)).head())
x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.2,random_state=42)
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
model=LogisticRegression(random_state=42,multi_class='ovr')
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(f"accuracy :{accuracy_score(y_test,pred)*100} %")
print(f"confusion matrix:{confusion_matrix(y_test,pred)}")
output=pd.DataFrame(pred)
expected=pd.DataFrame(y_test)
train_csv=pd.DataFrame(y_train)
df=pd.concat([output,expected],axis=1)
df.columns=["output","expected"]
print(df.head())