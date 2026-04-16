import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("predictive_maintenance.csv")

print(df.isnull().sum())

df = df.drop("UDI", axis =1)
df = df.drop("Product ID", axis =1)
df = df.drop("Failure Type", axis =1)

numeric_cols = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]

""" # Нормализация числовых данных
age_scaler = MinMaxScaler(feature_range=(300,))
df["numeric_cols"] = age_scaler.fit_transform(df[["numeric_cols"]]) """

# One-Hot Encoding категорий
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

df.to_csv("processed_predictive_maintenance.csv")

X = df.drop("Target", axis=1)
y=df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

DTC = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)
DTC.fit(X_train,y_train)
y_pred_test = DTC.predict(X_test)

print(accuracy_score(y_test,y_pred_test))

y_proba = DTC.predict_proba(X_test)
fpr,tpr,_ = roc_curve(y_test, y_proba[:,1])
auc_ss = auc(fpr,tpr)

plt.plot(fpr,tpr, color="red")
plt.plot([0,1],[0,1],color="orange")
plt.savefig("test.png")