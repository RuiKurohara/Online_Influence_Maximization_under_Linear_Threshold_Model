import pickle

def read_pickle_file(file_path):
    """
    Pickleファイルを読み込む関数。
    
    :param file_path: 読み込むファイルのパス
    :return: デシリアライズされたオブジェクト
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def process_data(data):
    """
    デシリアライズされたデータを処理し、テキストとして保存する関数。
    
    :param data: デシリアライズされたデータ
    """
    output_path = 'output.txt'
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            for key, value in data.items():
                line = f"キー: {key}, 値: {value}\n"
                file.write(line)
        print(f"データが {output_path} に保存されました。")
    except Exception as e:
        print(f"ファイルの書き込み中にエラーが発生しました: {e}")

def main():
    file_path = 'ER_node9_p_0.2EWTrue.dic'
    data = read_pickle_file(file_path)
    if data is not None:
        process_data(data)

if __name__ == "__main__":
    main()
