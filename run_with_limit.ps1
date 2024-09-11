# run_with_limit.ps1

# 実行回数を指定
$max_runs = 5  # 実行回数を変更できます
$wait_time = 5 # 次の実行までの待機時間（秒単位）

for ($i = 1; $i -le $max_runs; $i++) {
    Write-Host "Run #$i"
    
    # Pythonスクリプトを実行
    python Main.py --seed_size 3 --iterationTimes 500 --save_address SimulationResults/gaussian_25_celfpp_mc5_ER --G_address Datasets/ER_node25_p_0.2.G --weight_address Datasets/ER_node25_p_0.2EWTrue.dic --use_new_algorithm
    
    Write-Host "Run #$i finished."
    
    if ($i -lt $max_runs) {
        Write-Host "Waiting for $wait_time seconds before the next run..."
        Start-Sleep -Seconds $wait_time
    }
}

Write-Host "All $max_runs runs completed."
