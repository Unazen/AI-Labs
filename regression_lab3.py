import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error,r2_score 
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


#загрузка датасета
df = pd.read_csv("processed_regression.csv")
print(df.head())

#разделение на задачу и цели
X = df.drop("final_grade",axis=1)
y=df["final_grade"]

#разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#модель
dt_regressor_model = DecisionTreeRegressor(max_depth=8)
dt_regressor_model.fit(X_train, y_train)
y_pred_test=dt_regressor_model.predict(X_test)

#оценка
MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

#результаты
print("Mean Square Error, среднеквадратичная ошибка",MSE)
print("Root MSE, корень среднеквадратичной ошибки",RMSE)
print("Mean Absolute Error, средняя абсолютная ошибка",MAE)
print("r^2",r2)

# Визуализация дерева
plt.figure(figsize=(20, 10))
tree.plot_tree(dt_regressor_model, 
               feature_names=X.columns.tolist(), 
               filled=True, 
               fontsize=10,
               max_depth=4) # Ограничиваем отрисовку для красоты, даже если дерево глубже
plt.title("Структура дерева решений")
plt.savefig("regr_tree2.png")
plt.show()

""" 
y_pred_test.plot_tree(dt_regressor_model)
plt.savefig("regr_tree.png")
plt.show() """