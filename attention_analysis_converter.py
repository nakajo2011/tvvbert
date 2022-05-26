# -*- coding: utf-8 -*-
import os
import logging
import random
import json
import csv
from dataclasses import dataclass, field

import torch
import pandas as pd
from transformers import (
    BertTokenizerFast,
    AutoModelForMaskedLM,
    HfArgumentParser,
)


class AnalysisJsonConverter(object):
    """
    attention_model_to_json.pyで生成したAttentionの抽出データのsortid部分を番組のメタ情報に変換するためのクラス
    """

    def __init__(
            self,
            analysis_json_file=None,
            tv_meta_data_file=None,
            special_tokens = {'[CLS]', '[SEP]', '[MASK]'}
    ) -> None:
        """

        Parameters
        ----------
        analysis_json_file : str
            変換するjsonファイルのパス

        tv_meta_data_file : str
            番組のメタ情報のcsvファイルのパス
        """

        assert(analysis_json_file is not None), 'analysis_json_file not specify.'
        assert(tv_meta_data_file is not None), 'tv_meta_data_file not specify.'

        self.analysis_json = json.load(open(analysis_json_file))
        # 番組メタ情報をロード
        self.meta_df = pd.read_csv(tv_meta_data_file)
        self.special_tokens = special_tokens

    def convert(self) -> dict:
        """
        analysis.jsonのsortid部分を番組情報に変換する

        Returns
        -------
            番組情報に変換した、Attentionの関連度
        """
        result = []
        for item in self.analysis_json:
            atten_weights = []
            for sortid, atten_weight in item.items():
                meta = self._convert_sortid2meta_info(sortid)
                atten_weights.append([meta, atten_weight])
            result.append(atten_weights)
        return result

    def _convert_sortid2meta_info(self, sortid=None):
        """
        sortidを番組の放送時間、放送局、番組名、サブタイトルに変換して返す。
        渡されたsortidがspecial token([CLS]、[SEP]、[MASK])の場合はそのまま返す。

        Parameters
        ----------
        sortid: str
            番組のコマを表すid もしくは special token

        metadata_df: DataFrame
            番組のメタ情報

        Returns
        -------
            番組の放送時間、放送局、番組名、サブタイトルを結合した文字列

        """

        if sortid in self.special_tokens:
            return sortid

        target_df = self.meta_df.query(f'sortid == {sortid}')
        data_list = list(target_df[['from', '放送局', '番組名', 'サブタイトル']].values[0])
        return ' '.join([sortid, *map(str, data_list)])


@dataclass
class DataArguments:
    """
    コマンドライン引数を管理するクラス
    """

    analysis_json_file: str = field(
        metadata={'help': 'analysis.jsonファイルのパス'}
    )
    meta_data_file: str = field(
        metadata={'help': '視聴番組のメタデータファイル(csv形式)'},
    )
    output_dir: str = field(
        metadata={'help': '変換した後のデータを出力するフォルダのパス'},
    )


def main():
    """
    Usage:
        python3 attention_analysis_converter --analysis_json_file ./analysis.json --meta_data_file ./tv_meta_data.csv --output_dir ./output/
    Parameters
    ----------
    必須: attention_model_to_json.pyで出力したanalysis.json
    必須: 番組のメタ情報ファイル(csv形式)
    出力: sortid部分を番組情報に変換したデータ(csv形式)
        出力は１ユーザごとに output_dirで指定されてフォルダ以下にoutput_converted_{index}.csの名前のファイルに出力される。

    Returns
    -------

    """
    parser = HfArgumentParser(DataArguments)
    (data_args,) = parser.parse_args_into_dataclasses()
    converter = AnalysisJsonConverter(
        analysis_json_file=data_args.analysis_json_file,
        tv_meta_data_file=data_args.meta_data_file
    )
    # 出力フォルダを生成
    os.makedirs(data_args.output_dir, exist_ok=True)

    for i, converted in enumerate(converter.convert()):
        filepath = os.path.join(data_args.output_dir, f'output_converted_{i+1}.csv')
        f = open(filepath, 'w')
        w = csv.writer(f)
        for row in converted:
            w.writerow(row)


if __name__ == '__main__':
    main()
