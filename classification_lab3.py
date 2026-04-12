import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, auc, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
# import numpy as np
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
dt_classifier_model = DecisionTreeClassifier(max_depth=6)
dt_classifier_model.fit(X_train, y_train)

y_pred_test=dt_classifier_model.predict(X_test)

#работа с ROC кривой
#вероятности (матрица 16 колонок)
y_proba = dt_classifier_model.predict_proba(X_test)
print(dt_classifier_model.classes_)

""" #ROC кривая (только для бинарных категорий)
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
auc_metric = auc(fpr, tpr) """

# 2. Выбираем класс, который хотим проверить (например, под индексом 0)
class_index = 0
class_name = dt_classifier_model.classes_[class_index]

# 3. Создаем временную метку: 1 если это наш класс, 0 если любой другой из 15
y_test_binary = (y_test == class_name).astype(int)

# 4. Считаем ROC
fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, class_index])
roc_auc = auc(fpr, tpr)
score = roc_auc_score(y_test, y_proba, multi_class='ovr')

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
print("\n Roc Auc Score",score)

""" #Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.savefig("ROC-curve.png")
plt.show() """

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Список конфигураций для сравнения
configs = [
    {"name": "Без ограничений", "params": {}},
    {"name": "Глубина 3 (Pre-pruning)", "params": {"max_depth": 3}},
    {"name": "Глубина 6 (Pre-pruning)", "params": {"max_depth": 6}},
    {"name": "Мин. объектов в листе 10", "params": {"min_samples_leaf": 10}},
    {"name": "Обрезка (Post-pruning)", "params": {"ccp_alpha": 0.02}}
]

results = []

for config in configs:
    # Создаем и обучаем модель
    model = DecisionTreeClassifier(**config["params"], random_state=42)
    model.fit(X_train, y_train)
    
    # Считаем точность
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    results.append({
        "Вариант": config["name"],
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc
    })

# Выводим красивую таблицу
results_df = pd.DataFrame(results)
print(results_df)