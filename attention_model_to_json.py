import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Generator, List, Optional

from transformers import (BertForSequenceClassification, BertTokenizer,
                          HfArgumentParser)


class ExplainableBertModel(object):
    """
    BERTモデルの内部可視化を行うクラス

    Attributes
    ----------
    sentence_file: str
        文章ファイル
    model : BertForSequenceClassification
        BERTモデル
    tokenizer : BertTokenizer
        モデルに対応するトークナイザ
    """

    def __init__(self, sentence_file: str, data_dir: str, tokenizer_dir: str = None) -> None:
        """
        Parameters
        ----------
        sentence_file : str
            文章ファイル
        data_dir : str
            BERTモデル・トークナイザのディレクトリパス
        tokenizer_dir : str, default None
            トークナイザのディレクトリパス
        """
        if tokenizer_dir is None:
            tokenizer_dir = data_dir

        self.sentence_file = sentence_file
        self.model = self._load_model(data_dir)
        self.tokenizer = self._load_tokenizer(tokenizer_dir)

    def _load_model(self, model_dir: str) -> BertForSequenceClassification:
        """
        対象のディレクトリからBERTモデルを読み込む

        Parameters
        ----------
        model_dir : str
            モデルのディレクトリパス

        Returns
        -------
        model : BertForSequenceClassification
            読み込んだBERTモデル

        ------
        ValueError
            読み込み失敗時
        """
        try:
            model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=2,
                output_attentions=True
            )
        except ValueError as err:
            sys.stderr.write('Error: モデルの読み込みに失敗しました。')
            sys.stderr.write(err)

        return model

    def _load_tokenizer(self, tokenizer_dir: str) -> BertTokenizer:
        """
        対象のディレクトリからトークナイザを読み込む

        Parameters
        ----------
        model_dir : str
            トークナイザのディレクトリパス

        Returns
        -------
        model : BertForSequenceClassification
            読み込んだトークナイザ

        Raises
        ------
        ValueError
            読み込み失敗時
        """
        try:
            tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        except ValueError as err:
            sys.stderr.write('Error: トークナイザの読み込みに失敗しました。')
            sys.stderr.write(err)

        return tokenizer

    def _file_generator(self, file_path: str, mode: str = 'r') -> Generator[str, None, None]:
        """
        対象ファイルから1行毎に読み出すジェネレータを生成する

        Parameters
        ----------
        sentence_file : str
            ファイルパス
        mode : str
            ファイルを開くモード

        Returns
        -------
        line : str
            読み込んだ行
        """
        with open(file_path, mode) as file:
            for line in iter(file.readline, ''):
                yield line.rstrip(os.linesep)

    def tokenize(self, sentence_text: str) -> dict:
        """
        文章をトークナイズする。

        Parameters
        ----------
        sentence_text : str
            文章文字列

        Returns
        -------
        token : dict
            トークナイズしたデータ
        """
        return self.tokenizer.encode_plus(
            sentence_text,
            return_tensors='pt',
            padding="max_length",
            max_length=self.model.config.max_position_embeddings,
            truncation=True
        )

    def model_attentions(self, sentence_text: str) -> tuple:
        """
        モデルからAttentionの情報を取得する

        Parameters
        ----------
        sentence_text : str
            文章文字列

        Returns
        -------
        attentions : tuple
            モデルの全Attention
        """
        token = self.tokenize(sentence_text)
        model_out = self.model(token['input_ids'])

        return model_out['attentions']

    def is_truth(self, sentence_text: str, supervised=0) -> bool:
        """
        モデルの予測が正解しているかどうか

        Parameters
        ----------
        sentence_text : str
            文章文字列

        supervised : int
            正解のラベルID

        Returns
        -------
        attentions : bool
            予測が正解していたらTrue, 不正解の場合はFalse
        """
        token = self.tokenize(sentence_text)
        model_out = self.model(token['input_ids'])

        return model_out.logits.argmax(-1)[0] == supervised

    def attentions_to_dict(self, attentions: tuple, sentence_text: str) -> dict:
        """
        Attentionを辞書化する

        Parameters
        ----------
        attentions : tuple
            辞書化するAttention

        sentence_text : str
            文章文字列

        Returns
        -------
        attentions : dict
            モデルの全Attention
        """
        input_word_list = f'[CLS] {sentence_text} [SEP]'.split()
        response = {}
        for layer_index, layer in enumerate(attentions):
            layer_dict = {}
            for header_index, header in enumerate(layer[0]):
                header_dict = {}
                for key, value in zip(input_word_list, header):
                    header_dict[key] = dict(zip(input_word_list, value.tolist()))
                layer_dict[f'header{header_index+1}'] = header_dict
            response[f'layer{layer_index+1}'] = layer_dict

        return response

    def generate_data(self,
                      output_type: str = 'CLS',
                      total: int = 10,
                      supervised: int = 0
                      ) -> List[dict]:
        """
        指定されたフォーマットのデータを生成する。

        Parameters
        ----------
        output_type : str
            出力する辞書のフォーマット
            CLS(デフォルト) : Attentionの最終層を[CLS]トークンのみ集計した値
            WORDS : Attentionの最終層を各単語別に集計した値
            ALL : 全ての層の全Attention

        total : int, default 10
            出力するデータの総数。この数のユーザの視聴データに対するAttentionを出力する

        supervised : int, default 0
            入力されたデータの正解ラベル。
            BERTの予測結果がこのラベルIDと同じAttentionのみを出力する。

        Returns
        -------
        response : List[dict]
            生成した辞書
        """
        response = []
        for sentence_text in self._file_generator(self.sentence_file):
            if not self.is_truth(sentence_text, supervised=supervised):
                continue
            attentions = self.model_attentions(sentence_text)
            if output_type == 'CLS':
                attentions_dict = self.attentions_to_dict((attentions[-1],), sentence_text)
                data = self.aggregate_headers(attentions_dict['layer1'])['[CLS]']
            elif output_type == 'WORDS':
                attentions_dict = self.attentions_to_dict((attentions[-1],), sentence_text)
                data = self.aggregate_headers(attentions_dict['layer1'])
            elif output_type == 'ALL':
                data = self.attentions_to_dict(attentions, sentence_text)
            else:
                raise ValueError(f'不明な出力フォーマットです。 output_type:{output_type}')
            response.append(data)

            if len(response) == total:
                break

        return response

    def aggregate_headers(self, headers: dict, is_sort: bool = True, reverse: bool = True) -> dict:
        """
        Attentionヘッダーを集計する

        Parameters
        ----------
        headers : dict
            集計するAttentionヘッダー

        response : dict
            集計結果

        sort : dict, default True
            値順にソートするか

        reverse : dict, default True
            降順でソートするか

        Returns
        -------
        attentions : dict
            モデルの全Attention
        """
        # 各ヘッダーの単語別確率を1つに加算する
        counts = Counter({k: Counter() for k in headers['header1'].keys()})
        for header_dict in headers.values():
            for key, value in header_dict.items():
                counts[key] += Counter(value)

        # まとめた確率をヘッダー数で除算して正規化する
        response = {}
        header_num = len(headers)
        for key, value in counts.items():
            if is_sort:
                value = dict(sorted(value.items(), key=lambda i: i[1], reverse=reverse))
            response[key] = {k: (v / header_num) for k, v in value.items()}

        return response


@dataclass
class ScriptArguments(object):
    """
    mainスクリプト動作時の入出力引数を定義するデータクラス
    """
    sentence_file: str = field(
        metadata={'help': 'ユーザの視聴文章データファイル(.txt)'},
    )
    data_dir: str = field(
        metadata={
            'help': 'BERTモデル・トークナイザのディレクトリ'
        },
    )
    tokenizer_dir: Optional[str] = field(
        default=None,
        metadata={
            'help': 'トークナイザのディレクトリ(オプション: 指定する場合はdata_dirにモデルのディレクトリを指定してください)'
        },
    )
    output_file: str = field(
        default='analysis.json',
        metadata={
            'help': '出力するファイルのフルパス(デフォルト: ./analysis.json)'
        },
    )
    output_type: str = field(
        default='CLS',
        metadata={
            'help': '出力する辞書のフォーマット'
            'CLS(デフォルト): Attentionの最終層を[CLS]トークンのみ集計した値'
            'WORDS: Attentionの最終層を各単語別に集計した値'
            'ALL: 全ての層の全Attention'
        },
    )
    max_output_data: int = field(
        default=10,
        metadata={'help': '出力データの最大数'}
    )
    supervised_id: int = field(
        default=0,
        metadata={'help': '入力したデータの正解ラベルID'}
    )


def main():
    """
    転移学習したBERTモデルのattentionをjson形式で出力する。

    Parameters
    ----------
    必須: ユーザの視聴文章データファイル(.txt)
    必須: BERTモデルのディレクトリ
    任意: 出力するファイル(デフォルト: analysis.json)
    任意: 出力する辞書のフォーマット
            CLS(デフォルト) : Attentionの最終層を[CLS]トークンのみ集計した値
            WORDS : Attentionの最終層を各単語別に集計した値
            ALL : 全ての層の全Attention

    Usage
    --------
    python attention_model_to_json.py --sentence_file SENTENCE_FILE \
                                      --data_dir DATA_DIR \
                                      [--output_file OUTPUT_FILE]\
                                      [--output_type OUTPUT_TYPE]

    Examples
    --------
    python attention_model_to_json.py --sentence_file sentence.txt --data_dir classify_task/models
    python attention_model_to_json.py --sentence_file sentence.txt --data_dir classify_task/models \
     --output_file words.json --output_type WORDS --max_output_data 100
    """
    parser = HfArgumentParser(ScriptArguments)
    (args, *_) = parser.parse_args_into_dataclasses()

    ebm = ExplainableBertModel(args.sentence_file, args.data_dir, args.tokenizer_dir)

    print('=== データ生成 開始 ===')
    data = ebm.generate_data(
        args.output_type,
        total=args.max_output_data,
        supervised=args.supervised_id
    )

    print('=== 出力 開始 ===')
    with open(args.output_file, 'w') as file:
        file.write(json.dumps(data, indent=4))

    print('=== 完了 ===')


if __name__ == '__main__':
    main()
