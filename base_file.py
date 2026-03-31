#1import
import  pandas as pd
# train test
from sklearn.model_selection import train_test_split
# model selection (regression/classification)
from sklearn.linear_model import LinearRegression
#class
from sklearn.linear_model import LogisticRegression
#metrics (MSE, SMSE, MAE, r2)/(accurassy score, Cofusuion matrix, classification_report )
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#2read
# dr=pd.read_csv("processed_regression.csv")
dc=pd.read_csv("processed_classification.csv")

#3 target,param
""" X = dr.drop("final_grade", axis=1)
y=dr["final_grade"] 
 """
X=dc.drop("Personality", axis=1)
y=dc["Personality"]


#4 train test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)

#5 model

""" regress_model = LinearRegression()
regress_model.fit(X_train, y_train)
y_pred_test=regress_model.predict(X_test)  """

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_test=logistic_model.predict(X_test) 

#6 calc err

""" MSE = mean_squared_error(y_test, y_pred_test)
RMSE = root_mean_squared_error(y_test, y_pred_test)
MAE = mean_absolute_error(y_test, y_pred_test)
R2=r2_score(y_test, y_pred_test)
 """
AS = accuracy_score(y_test, y_pred_test)
CM = confusion_matrix(y_test, y_pred_test)
CR= classification_report(y_test, y_pred_test)

#7 print
""" print(MSE)
print(RMSE)
print(MAE)
print(R2) """

print(AS)
print(CM)
print(CR)

#Other things
#polynomical regerssion (uses linear as function)
from sklearn.preprocessing import PolynomialFeatures
n=9
poly_features = PolynomialFeatures(n)

X_train = poly_features.fit_transform(X_train)
linear_model.fit(X_train, y_train)


#model corection (L1 L2)
# l1 = |w|*lambda


from sklearn.linear_model import Lasso
l1 = Lasso(alpha=2.0)
l1.fit(X_train,y_train)
y_prob_test = l1.predict(X_test)

# l2 = w^2*lambda
from sklearn.linear_model import Ridge
l2 = Ridge(alpha=2.0)
l2.fit(X_train,y_train)
y_prob_test = l2.predict(X_test)