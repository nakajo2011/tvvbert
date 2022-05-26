# -*- coding: utf-8 -*-
import logging
import random
from dataclasses import dataclass, field
from typing import Generator, List, Tuple

import pandas as pd
import torch
from transformers import (AutoModelForMaskedLM, BertTokenizerFast,
                          HfArgumentParser)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments(object):
    """
    学習済みmodelのディレクトリなどの引数を管理するクラス
    """

    model_dir: str = field(
        metadata={'help': '事前学習済みのモデルが含まれたフォルダ'}
    )
    input_file: str = field(
        metadata={
            'help': '視聴データの文章データファイル'
            '穴あき問題のデータ。ロードされるのは先頭の行のみ。'
        },
    )
    meta_data_file: str = field(
        metadata={'help': '視聴番組のメタデータファイル(csv形式)'},
    )
    mask_idx: int = field(
        default=5,
        metadata={'help': '穴あきにする単語の位置'},
    )
    random_mask_idx: bool = field(
        default=True,
        metadata={
            'help': 'MASKする位置をランダムにするフラグ'
            'Trueの場合は、mask_idxの指定は無視される。'
        }
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The word length of a sequence(and BERT input)."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "テストデータをシャッフルする時のランダムシード値"}
    )
    test_max_num: int = field(
        default=10,
        metadata={"help": "入力ファイルからテスト対象とする最大のデータの行数"}
    )


class MaskedLanguageModelChecker(object):
    """
    BERTの穴埋め問題を実施するクラス

    Attributes
    ----------
    meta_df: DataFrame
        番組メタ情報
    model : AutoModelForMaskedLM
        BERTモデル
    tokenizer : BertTokenizerFast
        モデルに対応するトークナイザ
    """

    def __init__(self, meta_file: str, data_dir: str, tokenizer_dir: str = None) -> None:
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

        # 番組メタ情報をロード
        self.meta_df = pd.read_csv(meta_file)

        # 訓練済みのモデルをロード
        self.model = AutoModelForMaskedLM.from_pretrained(data_dir)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

        # 学習でパラメータが変わらないようにチェックモードに変更
        self.model.eval()

        # GPUを使わない場合
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Using CPU.")
            self.model.to('cpu')

    def load_random_words(self,
                          input_file: str,
                          seed: int,
                          read_num: int,
                          max_words: int
                          ) -> Generator[str, None, None]:
        """
        ファイルから指定された行数をランダムに読み込んでトークナイズする

        Parameters
        ----------
        input_file: str
            対象ファイル
        seed: int
            seed値
        read_num: int
            読み込む行数
        max_words: int
            最大単語数

        Returns
        -------
            番組の放送時間、放送局、番組名、サブタイトルを結合した文字列
        """
        # seed値を更新
        random.seed(seed)
        # データをロード
        with open(input_file, 'r') as f:
            for text in random.sample(f.readlines(), read_num):
                # BERTに入力するための形式に変換
                words = self.tokenizer.tokenize(text, truncation=True, max_length=max_words)
                yield [*words]

    def convert_sortid_to_meta_info(self, sortid) -> str:
        """
        sortidを番組の放送時間、放送局、番組名、サブタイトルに変換して返す。
        渡されたsortidがspecial token([CLS]、[SEP]、[MASK], [UNK])の場合はそのまま返す。

        Parameters
        ----------
        sortid: str
            番組のコマを表すid もしくは special token

        Returns
        -------
            番組の放送時間、放送局、番組名、サブタイトルを結合した文字列
        """

        if sortid in ('[CLS]', '[SEP]', '[MASK]', '[UNK]'):
            return sortid

        target_df = self.meta_df.query(f'sortid == {sortid}')
        data_list = list(target_df[['from', '放送局', '番組名', 'サブタイトル']].values[0])
        return ' '.join((sortid, *map(str, data_list)))

    def masked_word_predict(self, words, k: int = 5) -> Tuple[List[str], List[str]]:
        """
        [MASK]の位置にある単語を予測する

        Parameters
        ----------
        words: List[str]
            単語の配列
        k: int
            ランキングする数

        Returns
        -------
        words: Tuple[List[str], List[float]]
            予想した単語一覧, 予想率
        """

        # 単語をテンソルに変換
        word_ids = self.tokenizer.convert_tokens_to_ids(words)
        word_tensor = torch.tensor([word_ids])

        # [MASK]位置に入る適切な単語を予測
        y = self.model(word_tensor)

        # 予想結果から降順でkの数だけ値を取得
        vals, max_ids = torch.topk(y[0][0][words.index('[MASK]')], k=k)

        # 単語に変換
        result_words = self.tokenizer.convert_ids_to_tokens(max_ids.tolist())
        return result_words, vals


def main():
    """
    mlm_result_check
    BERTの穴埋め問題を実施するタスク
    複数行を持つデータを渡した場合は、ランダムにシャッフルした上で、最初の10行分だけを実施する。
    シャッフル時のseedや対象とするデータ件数はそれぞれ、--seed, test_max_numで変更可能。

    Usage:
    python3 mlm_result_check.py --model_dir pretraining_data/models --input_file ./data/sentence.txt \
    --meta_data_file ./data/tv-meta-data.csv --mask_idx 5

    入力データの形式:
        16278189006 16278816004 16278819004 16279014006 16279017006 16279020006 16279023006
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser(DataTrainingArguments)
    (data_args, ) = parser.parse_args_into_dataclasses()

    mask_index = data_args.mask_idx
    mlm_checker = MaskedLanguageModelChecker(data_args.meta_data_file, data_args.model_dir)
    words_iter = mlm_checker.load_random_words(
        input_file=data_args.input_file,
        read_num=data_args.test_max_num,
        seed=data_args.seed,
        max_words=data_args.max_seq_length
    )

    for words in words_iter:
        if data_args.random_mask_idx:
            # ランダムなindexに変更
            mask_index = random.randint(0, len(words)-1)

        expectid = words[mask_index]
        words[mask_index] = "[MASK]"
        words = ['[CLS]', *words, '[SEP]']

        print('入力された視聴行動番組一覧')
        print('----------------------')
        for word in words:
            print(mlm_checker.convert_sortid_to_meta_info(word))

        result = mlm_checker.masked_word_predict(words)

        # 単語を番組情報に変換して表示
        print('')
        print('[MASK]の候補となる番組')
        print('----------------------')
        for word, val in zip(*result):
            print(f'{mlm_checker.convert_sortid_to_meta_info(word)}, predict: {val}')

        print('')
        print(f'正解の単語: {mlm_checker.convert_sortid_to_meta_info(expectid)}')
        print('---------- END ----------\n')


if __name__ == '__main__':
    main()
