import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,Lasso,Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Bank_Churn.csv")

print(df.isnull().sum())

df = df.drop("CustomerId",axis=1)
df = df.drop("Surname",axis=1)
df = df.drop("Geography",axis=1)
df = df.drop("Gender",axis=1)
# df = df.drop("",axis=1)
numeric = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
scaler = MinMaxScaler()
df[numeric] = scaler.fit_transform(df[numeric])

#target - exited
X = df.drop("Exited", axis=1)
y=df["Exited"]

#train test
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.4,random_state=42)

l1 = Lasso(alpha=2.0)
l1.fit(X_train, y_train)
y_pred_test = l1.predict(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

print("Score",accuracy_score(y_test,y_pred_test))
print("CM", confusion_matrix(y_test,y_pred_test))
print("Report",classification_report(y_test,y_pred_test))