from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def get_mae(max_leaf_nodes, predictors_train,target_train, predictors_val, target_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state =0)
    model.fit(predictors_train, target_train)
    predicted_values = model.predict(predictors_val)
    mae = mean_absolute_error(predicted_values, target_val)
    return mae
    
#Random Foreset absolute error
def get_mae_rf(predictors_train, target_train, predictors_val, target_val):
    model = RandomForestRegressor()
    model.fit(predictors_train, target_train)
    predicted_values = model.predict(predictors_val)
    mae_rf = mean_absolute_error(predicted_values, target_val)
    return mae_rf

#random forest predictions
def prediction(train_X, train_y, test_X):
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    predicted_values  = model.predict(test_X)
    return predicted_values

'''DATA FROM FILE'''
main_file_path = r"./train.csv"
data = pd.read_csv(main_file_path)
#print(data.describe())
#print(data.columns)

'''output i.e SalePrice'''
y = data.SalePrice
#y.describe()

'''Columns of interest i.e. Choosen Features'''
columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X = data[columns]
#X.describe()

'''Splitting the data into training data into training data and Validation data'''
#train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
'''
# Define model
predict_model = DecisionTreeRegressor()

#Fitting on the training data
# Fit model
predict_model.fit(train_X, train_y)

#Now for validation: Predict the values in the validation data
# get predicted prices on validation data
val_predictions = predict_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


for max_leaf_nodes in [5,50,500,5000]:
    print("Max leaf nodes",max_leaf_nodes,"Mean Absolute Error",get_mae(max_leaf_nodes,train_X, train_y, val_X, val_y))
    
print(get_mae_rf(train_X, train_y, val_X, val_y))'''


test_data = pd.read_csv(r"./test.csv")
test_X = test_data[columns]

predicted_prices = prediction(X,y,test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
