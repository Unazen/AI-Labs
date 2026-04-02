#1 import
import pandas as pd
#train/test
from sklearn.model_selection import train_test_split
# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
# metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#2import

dc = pd.read_csv("processed_classification.csv")
# dr = pd.read_csv("processed_regression.csv")

# 3target
X=dc.drop("Personality", axis=1)
y=dc["Personality"]

""" X=dr.drop("final_grade", axis=1)
y=dr["final_grade"] """
# 4 train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# 5model
DTC = DecisionTreeClassifier(max_depth=4,min_samples_leaf=10,min_samples_split=10)
DTC.fit(X_train,y_train)
y_pred_test = DTC.predict(X_test)

# ---
""" DTR=DecisionTreeRegressor(max_depth=4,min_samples_leaf=10,min_samples_split=10)
# DTR=DecisionTreeRegressor() # эта версия хуже
DTR.fit(X_train,y_train)
y_pred_test=DTR.predict(X_test) """
# 6 error
# 7 result print
print(accuracy_score(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))


# print(r2_score(y_test, y_pred_test))

# 8additions
#post pruning
""" # 1. Находим эффективные альфа (путь обрезки)
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 2. Обучаем дерево с конкретным коэффициентом обрезки
model = DecisionTreeClassifier(ccp_alpha=0.01) """

#ROC curve - массив точек для постороения кривой
# поиск вероятностей
y_proba = DTC.predict_proba(X_test)
#roc-auc-score - среднее значение для оценки
score = roc_auc_score(y_test, y_proba, multi_class='ovr')
# print(auc_metric)
print(score)


# чтобы заработала кривая, нужно склеить все классы и поставить на сравнение с другими
class_idx=0
class_name=DTC.classes_[class_idx]

y_test_binary = (y_test==class_name).astype(int)
y_score_one_class = y_proba[:, class_idx]

fpr, tpr, thresholds = roc_curve(y_test_binary, y_score_one_class)
# auc - посчитать площадь
auc_metric = auc(fpr,tpr)