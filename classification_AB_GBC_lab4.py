import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_curve, roc_auc_score
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

#загрузка датасета
df=pd.read_csv("datasets/processed_classification.csv")
print(df.head())

#разделение на цель и параметры
X=df.drop("Personality", axis=1)
y=df["Personality"]

# разделение на train test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

#модель AB
AB = AdaBoostClassifier(learning_rate=1,n_estimators=500)
""" AB = AdaBoostClassifier(learning_rate=0.5,n_estimators=1000)
AB = AdaBoostClassifier(learning_rate=0.1,n_estimators=5000) """
AB.fit(X_train, y_train)
y_pred_test_ab = AB.predict(X_test)

#Оценка классификационной модели 
accuracy = accuracy_score(y_test, y_pred_test_ab)
#matrix
cm = confusion_matrix(y_test, y_pred_test_ab)
#report
report = classification_report(y_test, y_pred_test_ab)
#результаты
print("\nТочность:", accuracy)
print("\nConfusion matrix",cm )
print("\nОтчет: ", report)

#работа с ROC кривой (AB)
#вероятности (матрица 16 колонок)
y_proba = AB.predict_proba(X_test)
print(AB.classes_)

# 2. Выбираем класс, который хотим проверить (например, под индексом 0)
class_index = 0
class_name = AB.classes_[class_index]

# 3. Создаем временную метку: 1 если это наш класс, 0 если любой другой из 15
y_test_binary = (y_test == class_name).astype(int)

# 4. Считаем ROC
fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_index])
roc_auc = auc(fpr, tpr)
score = roc_auc_score(y_test, y_proba, multi_class='ovr')

#Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.savefig("ROC-curve-AB.png")
plt.show()

#модель GBC

GBC = GradientBoostingClassifier()
GBC.fit(X_train,y_train)
y_pred_test_gbc = GBC.predict(X_test)

#Оценка классификационной модели 
accuracy = accuracy_score(y_test, y_pred_test_gbc)
#matrix
cm = confusion_matrix(y_test, y_pred_test_gbc)
#report
report = classification_report(y_test, y_pred_test_gbc)
#результаты
print("\nТочность:", accuracy)
print("\nConfusion matrix",cm )
print("\nОтчет: ", report)

#работа с ROC кривой (GBC)
#вероятности (матрица 16 колонок)
y_proba = GBC.predict_proba(X_test)
print(GBC.classes_)

# 2. Выбираем класс, который хотим проверить (например, под индексом 0)
class_index = 0
class_name = GBC.classes_[class_index]

# 3. Создаем временную метку: 1 если это наш класс, 0 если любой другой из 15
y_test_binary = (y_test == class_name).astype(int)

# 4. Считаем ROC
fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_index])
roc_auc = auc(fpr, tpr)
score = roc_auc_score(y_test, y_proba, multi_class='ovr')

#Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.savefig("ROC-curve-GBC.png")
plt.show()