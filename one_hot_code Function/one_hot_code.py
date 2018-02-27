import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer


def get_mae(X,y):
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#drop the data with no target value i.e SalePrice
train_data.dropna(axis = 0, subset= ['SalePrice'], inplace =True)

target = train_data.SalePrice

col_with_missing_values = [col for col in train_data.columns
                                  if train_data[col].isnull().any()]

candidate_train_predictors = train_data.drop(['Id', 'SalePrice']+col_with_missing_values,axis=1)
candidate_test_predictors = test_data.drop(['Id'] + col_with_missing_values, axis = 1)


'''low_cardinality_cols = [cname for cname in candidate_train_predictors.columns
                                   if candidate_train_predictors[cname].nunique() < 10 and
                                    candidate_train_predictors[cname].dtype == "object"]'''

numeric_cols = [cname for cname in candidate_train_predictors.columns
                       if candidate_train_predictors[cname].dtype in ['int64','float64']]

my_columns =  numeric_cols #+low_cardinality_cols 

train_predictors  = candidate_train_predictors[my_columns]
test_predictors = candidate_test_predictors[my_columns]


one_hot_code_training_predictors = pd.get_dummies(train_predictors)
one_hot_code_testing_predictors = pd.get_dummies(test_predictors)
my_imputer = Imputer()

one_hot_code_training_predictors_imputed = my_imputer.fit_transform(one_hot_code_training_predictors)
one_hot_code_testing_predictors_imputed = my_imputer.transform(one_hot_code_testing_predictors)

'''final_train, final_test = one_hot_code_training_predictors.align(one_hot_code_testing_predictors, 
                                                                 join='left', axis = 1)'''



predictors_without_categories = train_data.select_dtypes(exclude =['object'])

predictors_without_categories_imputed = my_imputer.fit_transform(predictors_without_categories)
mae_without_categories = get_mae(predictors_without_categories_imputed,target)
mae_with_categories = get_mae(one_hot_code_training_predictors_imputed, target)

print(mae_without_categories)
print(mae_with_categories)




