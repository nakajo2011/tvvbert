import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Iterator

import pandas as pd
from transformers import HfArgumentParser


class SentenceConverter(object):
    """
    視聴データを文章データに変換するクラス
    """

    def convert_data(self, target_file: str, chunksize: int = 100000) -> Iterator[str]:
        """
        視聴データの変換を行う

        Parameters
        ----------
        target_file : str
            対象ファイル
        chunksize : int, default 100000
            ファイルから一回の読み込みで取得する行数

        Returns
        -------
        text_iter : iter
            読み込んだBERTモデル
        """
        sentences_dict = defaultdict(list)
        for df in pd.read_csv(target_file, chunksize=chunksize):
            self.categorize_user_id(df, sentences_dict)
            print(f'\r{datetime.now()}: {len(sentences_dict)} users data processed.', end='')
            sys.stdout.flush()
        print()

        # ユーザ毎にsort_idsをソート
        self.sort_sort_id(sentences_dict)

        # ユーザ毎の文章のリストに変換
        return self.replace_words(sentences_dict)

    def categorize_user_id(self, data_df: pd.DataFrame, sentences_dict: dict) -> None:
        """
        ユーザ毎にsort_idを配列にまとめる関数。
        変換結果は引数で渡された辞書に保管されます。
        結果の保管先を引数で渡しているのは、一度にファイル全体を読み込まずに逐次処理するためです。

        変換した後の辞書のデータ形式は以下の通り:
        sentences = {
            'user_id1': [sortid1, sortid2, ...],
            'user_id2': [sortid1, sortid2, ...],
            ...
            }

        Parameters
        ----------
        data_df : DataFrame
            csv形式の視聴データをロードしたDataFrame
        sentences_dict : defaultdict
            結果を格納する辞書
        """

        for _, user_id, sort_id in data_df.itertuples():
            sentences_dict[user_id].append(sort_id)

    def sort_sort_id(self, sentences_dict: defaultdict) -> None:
        """
        ユーザ毎にsort_idをソートする。
        引数で渡されたsentences内のsortidの配列に対して直接ソートする破壊的メソッド

        Parameters
        ----------
        sentences_dict : defaultdict
            ユーザ毎に振り分けられたsort_idsの辞書オブジェクト
        """

        for sort_ids in sentences_dict.values():
            sort_ids.sort()

    def replace_words(self, sentences) -> Generator[str, None, None]:
        """
        ユーザ毎のsort_idsをスペース区切りの文字に連結したテキストのジェネレーターを生成する

        Parameters
        ----------
        sentences : dict
            ユーザ毎に振り分けられたsort_idsの辞書オブジェクト

        Returns
        -------
        result : str
            スペース区切りの文字列
        """

        for sort_ids in sentences.values():
            yield ' '.join(map(str, sort_ids))


@dataclass
class ConvertingArguments(object):
    """
    視聴データを含むファイルや出力する文章データを含むファイルの引数を管理するクラス
    """
    csv_file: str = field(
        metadata={'help': '視聴データを含むファイル(csv形式)'}
    )
    sentence_file: str = field(
        metadata={'help': '出力する文章データを含むファイル(txt形式)'}
    )


def main():
    """
    convert-to-sentence

    視聴データを文章データに変換してtxt形式のファイルに出力するプログラム

    Usage:
    python3 convert_to_sentence.py --csv_file data/input.csv --sentence_file data/output.txt

    Data format:
    Input(csv_file):
    ```
    userid,sortid
    000001,16277766006
    000001,16277769006
    000001,16277790006
    000002,16278591005
    000002,16278729005
    ```

    Output(sentence_file):
    ```
    16277766006 16277769006 16277790006
    16278591005 16278729005
    ```
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser(ConvertingArguments)
    (converting_args, ) = parser.parse_args_into_dataclasses()

    converter = SentenceConverter()

    data_iter = converter.convert_data(converting_args.csv_file)

    # ユーザ毎の文章をファイルに保存する
    print(f'{datetime.now()}: write data to file.....')
    with open(converting_args.sentence_file, 'w') as f:
        # 速度を気にするならこれ
        # f.write('\n'.join(data_iter))
        # メモリを気にするならこれ
        for text in data_iter:
            f.write(f'{text}\n')

    print(f'{datetime.now()}: Generate sentence data Completed!!')


if __name__ == '__main__':
    main()
