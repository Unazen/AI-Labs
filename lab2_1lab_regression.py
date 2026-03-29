#импорт библиотек
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#открытие датасета
df = pd.read_csv("student_performance_prediction_dataset-2.csv")
print(df.head())

#Проверка на пустые значения
print(df.isnull().sum())

#выделение столбцов по параметрам
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns

#заполнение
""" for col in numeric_cols:
    df[col].fillna(df[col].median()) """

for col in categorical_cols:
    df[col]=df[col].fillna(df[col].mode()[0])

""" df["device_type"].fillna(df["device_type"].mode()[0])
df["extracurriculars"].fillna(df["extracurriculars"].mode()[0])
df["grade_category"].fillna(df["grade_category"].mode()[0]) """

print(df.isnull().sum())
#нормализация
""" scaler = MinMaxScaler() 
df[numeric_cols] = scaler.fit_transform(df[numeric_cols]) """

#ОНЕ
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.to_csv("processed_regression.csv", index=False)