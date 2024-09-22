# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect Data
Gather student data (e.g., marks, skills) and their placement status (placed or not placed).
2. Prepare Data
Convert data into numerical form (e.g., "placed" = 1, "not placed" = 0) and remove any missing values.
3. Train-Test Split
Split the data into training and testing sets.
4. Train Logistic Regression Model
Use the training data to train a logistic regression model.
5. Test and Predict
Test the model with the test data and use it to predict placement for new students. 

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.


Developed by: SARANYA S.


RegisterNumber: 212223220101


 ```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
print(data.head())
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([1,80,1,90,1,1,90,1,0,85,1,85])
```

## Output:
PREDICTED VALUE:

![image](https://github.com/user-attachments/assets/124cfe3c-c9a1-40c7-8464-c680662e7280)

ACCURACY:

![image](https://github.com/user-attachments/assets/766d1feb-6490-4adf-a774-6e77025f3433)

CONFUSION MATRIX:

![image](https://github.com/user-attachments/assets/78c73517-d9f7-486d-bf68-954576bf47a3)

CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/efca55af-488e-4e57-863f-5bad06b28119)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
