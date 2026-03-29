import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

#загрузка датасета
df=pd.read_csv("processed_classification.csv")
print(df.head())

#разделение на цель и параметры
X=df.drop("Personality", axis=1)
y=df["Personality"]

# разделение на train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

#модель
dt_classifier_model = DecisionTreeClassifier()


y_proba = ml_model.predict_proba(X_test)
print(ml_model.classes_)


fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
from sklearn.metrics import auc
auc_metric = auc(fpr, tpr)



#Оценка классификационной модели 
accuracy = accuracy_score(y_test, y_pred_test)

#matrix
cm = confusion_matrix(y_test, y_pred_test)

#report
report = classification_report(y_test, y_pred_test)

#результаты
print("\nТочность:", accuracy)
print("\nConfusion matrix",cm )
print("\nОтчет: ", report)