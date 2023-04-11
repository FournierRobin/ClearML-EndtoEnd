import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from clearml import Task
task = Task.init(project_name='House Price Regression', task_name='housereg_experiment')

df_train = pd.read_csv("data/cleaned_train.csv")
df_test = pd.read_csv("data/cleaned_test.csv")

X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

params = {'n_estimators': 100, 'learning_rate': 0.09}
task.connect(params)

gb = GradientBoostingRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
gb.fit(X_train, y_train)

task.upload_artifact('GBR model',gb)
joblib.dump(gb, 'models/gb.pkl', compress=True)

y_pred_gb = gb.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2 = r2_score(y_test, y_pred_gb)
task.get_logger().report_scalar("RMSE", "score", value=rmse, iteration=1)
task.get_logger().report_scalar("R2 score", "score", value=r2, iteration=1)





