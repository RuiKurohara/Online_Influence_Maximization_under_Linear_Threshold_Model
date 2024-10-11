import pandas as pd

# データ1とデータ2を読み込み
df1 = pd.read_csv('5_new.csv')
df2 = pd.read_csv('5.csv')

# 'Time(Iteration)'列を基準にしてデータを結合
merged_df = pd.merge(df1, df2, on='Time(Iteration)')

# 新しいCSVファイルとして保存
merged_df.to_csv('merged_5.csv', index=False)
