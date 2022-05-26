import os
from dataclasses import dataclass, field

import pandas as pd
from transformers import HfArgumentParser


def save_vocab(meta_data_file, dir_path, start_date='2021/08/01', end_date='2021/08/08'):
    """
    番組のsortidを単語として持つvocab.txtを作成する。
    引数で渡された番組メタデータから番組のsortidを抽出します。

    Parameters
    ----------
    meta_df : DataFrame
        csv形式の番組メタデータをロードしたDataFrame
    dir_path : str
        vocab.txtの出力先ディレクトリパスの文字列
    start_date: str
        vocab対象とするsortidの開始日時。放送日（開始）がこの日以降を対象とする。
    end_date: str
        vocab対象とするsortidの終了日時。放送日（開始）がこの日以前を対象とする。end_dateは含まない。
    """

    # 番組メタデータファイルを読み込む
    meta_df = pd.read_csv(meta_data_file)

    # 番組放送日（開始）　が文字コードエラーになるので、dateに置き換え
    rename_columns = list(meta_df.columns)
    rename_columns[3] = "date"
    meta_df.columns = rename_columns

    # 指定の日付範囲のデータだけに絞り込む
    target_term_df = meta_df.query(f'"{start_date}" <= date < "{end_date}"').sort_values('sortid')

    # スペシャルトークンを先頭に配置
    vocabs = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    for item in target_term_df.itertuples():
        vocabs.append(item.sortid)

    os.makedirs(dir_path, exist_ok=True)
    file = os.path.join(dir_path, 'vocab.txt')

    with open(file, 'w') as f:
        f.writelines('\n'.join(map(str, vocabs)))


@dataclass
class VocabGenerationArguments:
    """
    番組情報メタデータファイルとvocab.txtの出力先ディレクトリの引数を管理するクラス
    """

    meta_data_file: str = field(
        metadata={'help': '番組情報のメタデータファイル(csv形式)'}
    )
    output_dir: str = field(
        metadata={'help': 'vocab.txtの出力先ディレクトリ'}
    )


def main():
    """
    generate_vocab

    番組情報メタデータからvocab.txtを生成するプログラム

    Usage:
    python3 generate_vocab.py --meta_data_file data/tv_meta_data.csv --output_dir data/

    Output(sentence_file):
    ```
    [UNK]
    [CLS]
    [SEP]
    [PAD]
    [MASK]
    16277766006
    16277769006
    16277790006
    16278591005
    ...
    ```
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser(VocabGenerationArguments)
    (vocab_args, ) = parser.parse_args_into_dataclasses()

    # 番組メタデータからvocab.txtを出力する
    save_vocab(vocab_args.meta_data_file, vocab_args.output_dir)

    print('complete generate vocab.txt!')


if __name__ == '__main__':
    main()
