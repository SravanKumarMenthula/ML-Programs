import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('train.csv')
data.dropna(axis=0,subset = ['SalePrice'], inplace = True)

y= data.SalePrice
X = data.drop( ['SalePrice'],axis =1).select_dtypes(exclude = ['object'])

X_train, X_test, y_train, y_test  = train_test_split(X.as_matrix(), y.as_matrix(), test_size = 0.25)

my_imputer = Imputer()

X_train_impute = my_imputer.fit_transform(X_train)
X_test_impute = my_imputer.transform(X_test)

#1st model
my_model = XGBRegressor()

my_model.fit(X_train, y_train, verbose = False)

predictions_1 = my_model.predict(X_test)

print("Normal Error:" + str(mean_absolute_error(predictions_1, y_test)))

#2nd model

my_model = XGBRegressor(n_estimators =10000, learning_rate=0.05)

my_model.fit(X_train, y_train, early_stopping_rounds =5, eval_set =[(X_test, y_test)], verbose = False)

predictions_1  = my_model.predict(X_test)

print("Advance Error:" + str(mean_absolute_error(predictions_1, y_test)))



