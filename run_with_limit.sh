#!/bin/bash

# 実行回数を指定
max_runs=2  # 実行回数を変更できます
wait_time=5 # 次の実行までの待機時間（秒単位）
#seed_size=3 # シードサイズ
#iteration=10000 # 反復数
node_num=25 # ノード数
iterations=(1000 5000 10000) # 反復数
seed_sizes=(3 5) # シードサイズ

# シードサイズと反復数に基づいてPythonスクリプトを実行
for seed_size in "${seed_sizes[@]}"; do
    for iteration in "${iterations[@]}"; do
        for (( i=1; i<=$max_runs; i++ )); do
            echo "Run #$i"
            
            # Pythonスクリプトを実行
            command="python3 Main_optuna_simple.py --seed_size $seed_size --iterationTimes $iteration --save_address SimulationResults/AETC_optuna_pruning_simple/gaussian_${node_num}_AETC_${iteration}_k${seed_size}_ER --G_address Datasets/ER_node${node_num}_p_0.2.G --weight_address Datasets/ER_node${node_num}_p_0.2EWTrue.dic"
            
            # コマンド実行
            eval $command
            echo "Run #$i finished."
            
            # 次の実行まで待機
            if [ $i -lt $max_runs ]; then
                echo "Waiting for $wait_time seconds before the next run..."
                sleep $wait_time
            fi
        done
    done
done

echo "All $max_runs runs completed."
