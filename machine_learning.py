import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import xgboost as xgb

def search_model_params(x_train, y_train):
    params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000]}
    
    model = xgb.XGBRegressor(enable_categorical=True)
    search = GridSearchCV(estimator=model, param_grid=params,scoring='neg_mean_squared_error', verbose=1)
    search.fit(x_train, y_train)
    print("Best parameters:", search.best_params_)
    print("Lowest RMSE: ", (-search.best_score_)**(1/2.0))
      

def get_data(file):
    data = pd.read_csv(file)

    x = data.iloc[:, 1:-1] #данные
    y = data.iloc[:, -1] #результат

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.66, random_state = 70)

    categoty_attribs = list(x.select_dtypes(include=['object']).columns)
    x_train[categoty_attribs] = x_train[categoty_attribs].astype('category')
    x_test[categoty_attribs] = x_test[categoty_attribs].astype('category')
    
    return x_train, x_test, y_train, y_test


def model_training(x_train, x_test, y_train, y_test):

    params = {  "objective": "reg:squarederror",
                "n_estimators":500,
                "max_depth": 6,
                "learning_rate": 0.05
                }
    
    model = xgb.XGBRegressor(enable_categorical=True, **params)

    eval_set = [(x_test, y_test)]
    model.fit(x_train, y_train, eval_metric="mae", eval_set=eval_set, verbose=True, early_stopping_rounds=40)
    
    y_predict = model.predict(x_test)

    print('MAE: ', metrics.mean_absolute_error(y_predict, y_test))
    print('MSE: ', metrics.mean_squared_error(y_predict, y_test)) 
    print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_predict, y_test)))

    model.save_model('model.json')

def train():
    x_train, x_test, y_train, y_test = get_data('data.csv')
    model_training(x_train, x_test, y_train, y_test)

def get_answer(human_data, product_data):
    params = human_data | product_data
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model('model.json')

    params = pd.DataFrame(params, index=[0])
    categoty_attribs = list(params.select_dtypes(include=['object']).columns)
    params[categoty_attribs] = params[categoty_attribs].astype('category')

    answer = loaded_model.predict(params)
    return int(answer)

