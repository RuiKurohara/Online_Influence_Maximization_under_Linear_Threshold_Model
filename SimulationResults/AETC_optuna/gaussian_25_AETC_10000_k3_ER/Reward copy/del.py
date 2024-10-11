import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('5.csv')

# 'AETC_1'列を削除する
df = df.drop(columns=['AETC_1'])

# 新しいCSVファイルとして保存する
df.to_csv('5_new.csv', index=False)
