# run_with_limit.ps1

# 実行回数を指定
$max_runs = 5  # 実行回数を変更できます
$wait_time = 5 # 次の実行までの待機時間（秒単位）
#$seed_size = 3 #シードサイズ
#$iteration = 10000 #反復数
$node_num = 25 #ノード数
$iterations = @(1000,5000,10000) #反復数
$seed_sizes = @(3,5) #ノード数
foreach($seed_size in $seed_sizes){
    foreach($iteration in $iterations){
        for ($i = 1; $i -le $max_runs; $i++) {
            Write-Host "Run #$i"
            
            # Pythonスクリプトを実行
            $command = "python Main_optuna_simple.py --seed_size $seed_size --iterationTimes $iteration --save_address SimulationResults/AETC_optuna_simple2/gaussian_${node_num}_AETC_${iteration}_k${seed_size}_ER --G_address Datasets/ER_node${node_num}_p_0.2.G --weight_address Datasets/ER_node${node_num}_p_0.2EWTrue.dic"
            
            Invoke-Expression $command
            Write-Host "Run #$i finished."
            
            if ($i -lt $max_runs) {
                Write-Host "Waiting for $wait_time seconds before the next run..."
                Start-Sleep -Seconds $wait_time
            }
        }
    }
}
Write-Host "All $max_runs runs completed."

#.\run_with_limit.ps2