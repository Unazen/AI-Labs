# Импорт библиотек
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Загрузка датасета
df = pd.read_csv("data.csv")

print("Первые 5 строк:")
print(df.head())
print("\nИнформация о датасете:")
df.info()

#Проверка пропусков
print("\nКоличество пропусков:")
print(df.isnull().sum())
#Константы парметров
numeric_cols = [
    "Age",
    "Education",
    "Introversion Score",
    "Sensing Score",
    "Thinking Score",
    "Judging Score"
]

categorical_cols = ["Gender", "Interest"]

#заполнение пропусков
for col in numeric_cols:
    df[col]=df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col]=df[col].fillna(df[col].mode(), inplace=True)

# Нормализация числовых данных
age_scaler = MinMaxScaler(feature_range=(0,10))
df["Age"] = age_scaler.fit_transform(df[["Age"]])

# One-Hot Encoding категорий
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Проверка результата
print("\nПервые строки обработанных данных:")
print(df.head())

# Сохранение
df.to_csv("processed_lab1_classification.csv", index=False)