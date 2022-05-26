import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser


class ClassifyTrainingData(object):
    """
    文章データから文章分類のための教師データを作成するクラス

    Attributes
    ----------
    sort_ids : Tuple[int]
        ソートするID一覧
    """

    def __init__(self, data_file: str, tv_program_name: str, tv_program_subtitle: str) -> None:
        """
        Parameters
        ----------
        data_file : str
            対象のデータファイル
        tv_program_name : str
            対象番組の番組名
        tv_program_subtitle : str
            対象番組のサブタイトル
        """
        self.sort_ids = self.generate_sortid_list(data_file, tv_program_name, tv_program_subtitle)

    def categorize(self, data_file: str, categories: List[str]) -> None:
        """
        文章データを読み込み、分類ラベル毎にユーザを分類する

        Parameters
        ----------
        data_file : str
            対象のデータファイル
        categories : List[str]
            分類するラベル
        """
        # 分類したトレーニングデータの辞書の定義
        data_dict = {k: [] for k in categories}

        with open(data_file, 'r') as f:
            for line in tqdm(f):
                category, data = self.classify_user(list(map(int, line.split())), categories)
                data_dict[category].append(data)

        print(f'{datetime.now()}: {sum(map(len, data_dict.values()))} users data processed.')

        return data_dict

    def generate_sortid_list(self,
                             data_file: str,
                             tv_program_name: str,
                             subtitle: str = ''
                             ) -> Tuple[int]:
        """
        番組メタデータから視聴対象となる番組のsort_idのリストを生成する

        Parameters
        ----------
        tv_program_name : str
            番組名
        subtitle : string, default ''
            サブタイトル

        Returns
        -------
        sortids : Tuple
            視聴対象となる番組のsort_idのタプル
        """
        df = pd.read_csv(data_file)
        df = df[df['番組名'] == tv_program_name].sort_values('sortid')

        if subtitle:
            df = df[df['サブタイトル'] == subtitle]

        return tuple(map(int, df['sortid'].values))

    def classify_user(self,
                      words: List[str],
                      categories: List[str],
                      threshold: float = 0.333
                      ) -> Tuple[bool, str]:
        """
        指定されたtarget_sortidsをもとに、視聴した文章データと視聴していない文章データに分類する

        Parameters
        ----------
        words : List[str]
            文章データのリスト
        categories : List[str]
            分類するラベル
        threshold : float, default 0.333
            視聴したと判定する割合

        Returns
        -------
        result : Tuple(bool, str)
            判定結果、文章データ
        """

        # 文章毎に視聴している・していないで分類する
        # 視聴と判定する対象番組コマの合計数
        threshold_num = int(len(self.sort_ids) * threshold)
        # 0の場合に1件も視聴データがなくても視聴されたと誤判定されることを回避
        if threshold_num <= 0:
            threshold_num = 1

        # 教師データの範囲は、対象番組の直前の番組視聴データまで
        min_id = min(self.sort_ids)
        training_sentence = tuple(int(w) for w in words if int(w) < min_id)
        # 視聴合計数からthreshold以上視聴している・していないを判定
        is_view = (len(set(words) & set(self.sort_ids)) >= threshold_num)

        category = categories[0] if is_view else categories[1]

        return category, ' '.join(map(str, training_sentence))

    def replace_words_list(sentences):
        """
        ユーザ毎のsort_idsをスペース区切りの文字に連結したリストに変換する

        Parameters
        ----------
        sentences : dict
            ユーザ毎に振り分けられたsort_idsの辞書オブジェクト

        Returns
        -------
        result : list
            スペース区切りの文字列にまとめられたリスト
        """

        result = []
        for item in sentences:
            result.append(' '.join(map(str, iter(item))))

        return result


@ dataclass
class ClassifyingArguments(object):
    """
    文章データを含むファイルや分類した教師データの出力先ディレクトリの引数などを定義するデータクラス
    """
    sentences_data_file: str = field(
        metadata={'help': '文章データを含むファイル(txt形式)'}
    )
    tv_meta_data_file: str = field(
        metadata={'help': '番組メタデータを含むファイル(csv形式)'}
    )
    training_data_dir: str = field(
        metadata={'help': '分類した教師データの出力先ディレクトリ'}
    )


def main():
    """
    classify_training_data

    文章データから文章分類のための教師データを作成するプログラム

    「東京2020オリンピック, ◇野球　決勝「日本×アメリカ」（中継）」の番組のコマのうち、
    1/3以上のコマを視聴した人を「視聴した」人、そうでない人を「視聴していない」人として分類する。
    視聴した人・視聴していない人の文章データを、指定された出力先ディレクトリにそれぞれ別々のファイル(viewed.txt, not_viewed.txt)に保存する。
    教師データには対象番組より前の使用行動データ(文章)のみ出力する。

    Usage: python3 classify_training_data.py --sentences_data_file data/sentence.txt --tv_meta_data_file data/tv_meta_data.csv --training_data_dir data/training_data

    Data format:
    Input(sentences_data_file):
    ```
    16280769001 16281222004 16281225004 16281252004
    16277787006 16280697004 16283355001 16283409001 16283415001
    16279689008 16279872008 16281645004 16281666004 16282041004 16282542001 16282545005 16283067001 16283337001 16283418001 # noqa: E501
    ```

    Output(training_data):
    viewed.txt
    ```
    16277787006 16280697004
    ```

    not_viewed.txt
    ```
    16280769001 16281222004 16281225004 16281252004
    16279689008 16279872008 16281645004 16281666004 16282041004 16282542001 16282545005 16283067001
    ```
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser(ClassifyingArguments)
    (classifying_args, ) = parser.parse_args_into_dataclasses()

    tv_program_name = '東京2020オリンピック'
    tv_program_subtitle = '◇野球　決勝「日本×アメリカ」（中継）'
    categories = ['viewed_v2', 'not_viewed_v2']

    ctd = ClassifyTrainingData(
        classifying_args.tv_meta_data_file,
        tv_program_name,
        tv_program_subtitle
    )

    # データの分類を行う
    training_data = ctd.categorize(classifying_args.sentences_data_file, categories)
    print(f'{datetime.now()}: viewed {len(training_data[categories[0]])}'
          f' users and not_viewed {len(training_data[categories[1]])} users data processed.')

    # 分類した視聴している・していない文章をそれぞれファイルに保存する
    for key, value in training_data.items():
        print
        if not value:
            print(f'{key} training data not found')
            continue

        print(f'{datetime.now()}: write {key} training data to file.....')
        with open(os.path.join(classifying_args.training_data_dir, f'{key}.txt'), 'w') as f:
            f.write('\n'.join(value))


if __name__ == '__main__':
    main()
