import os
import optuna
import matplotlib.pyplot as plt

# 保存先のフォルダを指定


# フォルダが存在しない場合は作成


# 目的関数（例：x^2 + y^2を最大化）
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return x ** 2 + y ** 2

# スタディを作成
study = optuna.create_study(direction="minimize")

# 手動で特定のパラメータをキューに追加 (x=0.5, y=1)
for i in range(11):
    study.enqueue_trial({'x': 0.5, 'y': i-5})

# 通常の最適化プロセスを続行
study.optimize(objective, n_trials=100)

# 最終的な結果を確認
print("最良のパラメータ:", study.best_params)
print("最良の値:", study.best_value)

# プロットを表示
plt.plot([trial.value for trial in study.trials])
plt.grid()
save_dir = 'test'
# 'test' フォルダに画像を保存
plt.savefig(os.path.join(save_dir, 'optuna_plot.png'))  # フォルダパスを指定
#plt.show()
