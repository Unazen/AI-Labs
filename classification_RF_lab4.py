import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_curve
# from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

#загрузка датасета
df=pd.read_csv("processed_classification.csv")
print(df.head())

#разделение на цель и параметры
X=df.drop("Personality", axis=1)
y=df["Personality"]

# разделение на train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

#модель


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

#Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.savefig("ROC-curve.png")
plt.show()