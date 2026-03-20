import pandas as pd
from sklearn.metrics import roc_curve

df=pd.read_csv("processed_classification.csv")

y_proba = ml_model.predict_proba(X_test)
print(ml_model.classes_)


fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])