# 2次関数
import optuna
def f(x):
    return ((x - 3) ** 2)
# 目的関数の設定（ステップ1）
def objective(trial):
    x = trial.suggest_float('x', -10, 20)  #ハイパーパラメータの集合を定義する
    return f(x)                           #良し悪しを判断するメトリクスを返す
# 目的関数の最適化を実行する（ステップ2）
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
# 最適解の出力
print(f"The best value is : \n {study.best_value}")
print(f"The best parameters are : \n {study.best_params}")
