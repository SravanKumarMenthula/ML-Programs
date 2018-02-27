from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
import pandas as pd


'''DATA FROM FILE'''
main_file_path = '../input/train.csv'
data = pd.read_csv(main_file_path)

data.dropna(axis=0,subset = ['SalePrice'], inplace = True)

y_train= data.SalePrice

low_car_train = [col for col in data.columns
                       if data[col].nunique() < 10 and
                       data[col].dtype == 'object']

numeric_cols_train = [col for col in data.columns
                          if data[col].dtype in ['int64', 'float64']]

cols_trains = low_car_train + numeric_cols_train

temp_data = data[cols_trains]

X_train = temp_data.drop(['Id','SalePrice'],axis=1)
#X_train = data.drop( ['SalePrice'],axis =1).select_dtypes(exclude = ['object'])



#X_train, X_test, y_train, y_test  = train_test_split(X.as_matrix(), y.as_matrix(), test_size = 0.25)

test_data = pd.read_csv('../input/test.csv')

low_car_test = [col for col in test_data.columns
                       if test_data[col].nunique() < 10 and
                       test_data[col].dtype == 'object']

numeric_cols_test = [col for col in test_data.columns
                          if test_data[col].dtype in ['int64', 'float64']]

cols_test = low_car_test + numeric_cols_test

temp_data_test = test_data[cols_test]

X_test = temp_data_test.drop(['Id'],axis=1)
#X_test = test_data.select_dtypes(exclude = ['object'])

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align( X_test,  join='inner', axis=1)

my_imputer = Imputer()

X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.transform(X_test)

predicted_prices = XGBRegressor_predict(X_train,y_train,X_test)
print(predicted_prices)



my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
