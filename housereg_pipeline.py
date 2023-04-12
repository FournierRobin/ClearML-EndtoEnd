import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import clearml
from clearml import PipelineDecorator, PipelineController

@PipelineDecorator.component(cache=True, return_values=['df_train', 'df_test'])
def load_datasets():
    import pandas as pd
    df_train = pd.read_csv("data/cleaned_train.csv")
    df_test = pd.read_csv("data/cleaned_test.csv")
    return df_train, df_test

@PipelineDecorator.component(return_values=['X_train', 'X_test', 'y_train', 'y_test'])
def datasets_split(df_train):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    X = df_train.drop('SalePrice', axis=1)
    y = df_train['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(return_values=['gb'])
def train_model(X_train, y_train, loss='squared_error', learning_rate=0.1):
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate)
    gb.fit(X_train, y_train)
    return gb


@PipelineDecorator.component(return_values=['rmse', 'r2'])
def evaluate(X_test, y_test, model):
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    from clearml import Task
    task = Task.current_task()
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    task.get_logger().report_scalar("RMSE", "score", value=rmse, iteration=1)
    task.get_logger().report_scalar("R2 score", "score", value=r2, iteration=1)
    return rmse, r2

@PipelineDecorator.component(return_values=['y_pred', 'df_test'])
def predict(df_test, model):
    y_pred = model.predict(df_test)
    df_test['SalePrice_Prediction'] = y_pred
    return y_pred, df_test

@PipelineDecorator.pipeline(name="Full Pipeline", project="House Reg Pipeline", version=1.0)
def main():
    df_train, df_test = load_datasets()
    X_train, X_test, y_train, y_test = datasets_split(df_train)
    model = train_model(X_train, y_train)
    rmse, r2 = evaluate(X_test, y_test, model)
    predictions, predict_dataset = predict(df_test, model)

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    main()
