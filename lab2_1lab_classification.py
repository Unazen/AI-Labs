# Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 

# Загрузка датасета
df = pd.read_csv("processed_classification.csv")
print(df.head())

#разделение на цель и параметры
X=df.drop("Personality", axis=1)
y=df["Personality"]

# разделение на train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

#Логистическая регрессия
logreg_model = LogisticRegression(max_iter=10000)

logreg_model.fit(X_train, y_train)
y_pred_test = logreg_model.predict(X_test)

#Оценка классификационной модели 
accuracy = accuracy_score(y_test, y_pred_test)

#matrix
cm = confusion_matrix(y_test, y_pred_test)

""" #plt.ion
plt.figure(figsize=(16, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.show()
#plt.savefig("cm.png")
#print("График сохранён в cm.png") """

#report
report = classification_report(y_test, y_pred_test)

#результаты
print("\nТочность:", accuracy)
print("\nConfusion matrix",cm )
print("\nОтчет: ", report)