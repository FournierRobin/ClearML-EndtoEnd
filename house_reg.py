import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_datasets():
    df_train = pd.read_csv("data/cleaned_train.csv")
    df_test = pd.read_csv("data/cleaned_test.csv")
    return df_train, df_test

def datasets_split(df_train):
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, loss='squared_error', learning_rate=0.1):
    gb = GradientBoostingRegressor(loss, learning_rate)
    gb.fit(X_train, y_train)
    return gb

def evaluate(X_test, y_test, model):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def predict(df_test, model):
    y_pred = model.predict(df_test)
    df_test['SalePrice_Prediction'] = y_pred
    return y_pred, df_test

def main():
    df_train, df_test = load_datasets()
    X_train, X_test, y_train, y_test = datasets_split(df_train)
    model = train_model(X_train, y_train)
    rmse, r2 = evaluate(X_test, y_test, model)
    print('Root Mean Squared Error:', rmse)
    print('R2 Score:', r2)
    predictions, predictions_dataset = predict(df_test, model)

if __name__ == '__main__':
    main()
