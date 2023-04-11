from clearml.automation import UniformIntegerParameterRange, UniformParameterRange, DiscreteParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml import Task

Task.init(project_name='House Reg HPO', 
          task_name=f"optimizer", 
          task_type=Task.TaskTypes.optimizer)

optimizer = HyperParameterOptimizer(
    base_task_id='a0fed7fc364044be9df7b6760c91abc3',
    hyper_parameters=[
        UniformIntegerParameterRange('General/n_estimators', min_value=10, max_value=250),
        UniformParameterRange('General/learning_rate', min_value=0.001, max_value=0.1)
    ], 
    objective_metric_title='RMSE',
    objective_metric_series='score',
    objective_metric_sign='min',
    max_number_of_concurrent_tasks=10
)
optimizer.start()
optimizer.wait()
optimizer.stop()