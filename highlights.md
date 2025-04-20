# Highlights

1. Engineer successfully implements Optuna for tuning LightGBM model. 

<details>
<summary>Output from engineer</summary>
---------- ModelClientStreamingChunkEvent (engineer) ----------
OBSERVATION:
- The LightGBM + Optuna integration works successfully when using `lgb.early_stopping` callback in the `callbacks` keyword.
- 1-trial test ran successfully, yielding a validation MAE and early stopping at iteration 1274 (which also verifies that early stopping works).
- No `TypeError` or runtime exceptions encountered.

THOUGHT:
The hyperparameter optimization framework is now correct and ready for the full (100-trial) optimization run. Next, I will:
- Execute the 100-trial Optuna optimization using the callback-based early stopping.
- Save all tuning logs, study artifacts, final LightGBM model, and validation predictions, as detailed previously.
- This is a computationally intensive operation, but is justified by the SOTA procedure requirements, and should be manageable on current hardware for ~2k features.

ACTION:
```bash
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate sklearn-env
# --- Start Python code --- 
PYTHONFAULTHANDLER=1 python - <<END
import numpy as np
import lightgbm as lgb
import optuna
import pandas as pd
import joblib

X_train = np.load('task_7_iteration_3/X_train_matrix.npy')
y_train = np.load('task_7_iteration_3/y_train_array.npy')
X_val = np.load('task_7_iteration_3/X_val_matrix.npy')
y_val = np.load('task_7_iteration_3/y_val_array.npy')

def objective(trial):
    param = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0)
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dval],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    val_mae = np.mean(np.abs(val_pred - y_val))
    return val_mae

print("Starting Optuna hyperparameter optimization (n_trials=100) for LightGBM...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

best_params = study.best_trial.params
best_val_mae = study.best_value
print("Best val MAE:", best_val_mae)
print("Best hyperparameters:", best_params)

# Train final LGBM model with early stopping (record best iteration)
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)
final_gbm = lgb.train(
    {**best_params, "objective": "regression", "metric": "mae", "verbosity": -1, "boosting_type": "gbdt", "n_jobs": -1},
    dtrain,
    valid_sets=[dval],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=20)]
)
final_gbm.save_model("task_7_iteration_3/model_lgb.txt")
joblib.dump(final_gbm, "task_7_iteration_3/model_lgb.pkl")
joblib.dump(study, "task_7_iteration_3/optuna_study.pkl")
params_df = pd.DataFrame([trial.params for trial in study.trials])
params_df['val_mae'] = [trial.value for trial in study.trials]
params_df.to_csv("task_7_iteration_3/optuna_tuning_history.csv", index=False)
print("Final model and tuning artifacts saved.")

val_pred = final_gbm.predict(X_val, num_iteration=final_gbm.best_iteration)
np.save("task_7_iteration_3/pred_val_array.npy", val_pred)
END
# --- End Python code --- 
echo "Bash script finished."
```
[Prompt tokens: 0, Completion tokens: 0]
</details>




