# -*- coding: utf-8 -*-
import glob
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset, load_metric
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification,
                          BertTokenizer, HfArgumentParser, Trainer,
                          TrainingArguments)


logger = logging.getLogger(__name__)


@dataclass
class ClassifyTrainingConfg(object):
    """
    事前学習データと学習済みmodelの出力先ディレクトリなどの設定.
    """

    model_dir: str = field(
        metadata={'help': 'The dir of BERT pre-trained model config files.'}
    )
    train_dir: str = field(
        metadata={'help': 'The dir of training data files.'},
    )
    freeze_bert_params: bool = field(
        default=True,
        metadata={
            'help': '分類タスク学習時にBERTのパラメータが更新されないようにするかどうか。default True'
        },
    )
    use_checkpoint: bool = field(
        default=True,
        metadata={
            'help': 'checkpointから学習を再開するかどうか。default True'
                    '指定されたoutput_dirの中にあるcheckpointを利用する。checkpointが存在しない場合はこのフラグは無視される'
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            'help': '入力文書の単語数。1つの文章は必ずこの単語数になるように足りない部分は[PAD]で埋められる'
        }
    )
    train_data_max_per_label: int = field(
        default=10000,
        metadata={
            'help': 'labelごとの訓練データの最大数'
                    'labelごとの訓練データの最大数。この数までの訓練データをファイルからロードする。この数に満たない場合は全てのデータをロードする'
        }
    )
    randomize_data: bool = field(
        default=True,
        metadata={
            'help': '指定された教師データのランダムなデータを学習及び、予測に用いるかどうか'
        }
    )
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "データロード時の並列処理数。"
        }
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "キャッシュのデータを上書きするかどうか。"
                    "Falseの場合はキャッシュからデータをロードする。Trueの場合は新規にロードしなおす。"
        }
    )


class ClassifyTraining(object):
    """
    文章分類タスクをfine-tuningするクラス

    Attributes
    ----------
    training_config: DataTrainingArguments
        学習の設定情報
    categories: List[str]
        カデゴリラベル
    model: BertForSequenceClassification
        モデル
    tokenizer: BertTokenizer
        モデルに対応するトークナイザ
    trainer: Trainer
        トレーナー
    dataset_test: Dataset
        検証時に使用するデータセット
    """
    _param_sentence_key = 'sentence'
    _param_label_key = 'label'

    def __init__(self, training_config: ClassifyTrainingConfg,
                 categories: List[str],
                 data_dir: str,
                 tokenizer_dir: str = None
                 ) -> None:
        """
        Parameters
        ----------
        config : ClassifyTrainingConfg
            トレーニングの設定情報
        categories : List[str]
            カデゴリラベル
        data_dir : str
            BERTモデル・トークナイザのディレクトリパス
        tokenizer_dir : str, default None
            トークナイザのディレクトリパス
        """
        if tokenizer_dir is None:
            tokenizer_dir = data_dir

        self.training_config = training_config
        self.categories = categories
        self.tokenizer = self._load_tokenizer(tokenizer_dir)
        self.model = self._load_model(data_dir)
        self.trainer = None
        self.dataset_test = None
        self.metrics = load_metric('f1')

        # GPUを使わない場合
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Using CPU.")
            self.model.to('cpu')

    def _load_tokenizer(self, tokenizer_dir: str) -> BertTokenizer:
        """
        対象のディレクトリからトークナイザを読み込む

        Parameters
        ----------
        tokenizer_dir : str
            トークナイザのディレクトリパス

        Returns
        -------
        tokenizer : BertTokenizer
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

        Raises
        ------
        ValueError
            読み込み失敗時
        """
        try:
            # BERTのconfigファイルをロード
            config = BertConfig.from_json_file(
                os.path.join(model_dir, 'config.json'))
            # 分類ラベル数を指定
            config.num_labels = len(self.categories)
            model = BertForSequenceClassification.from_pretrained(
                os.path.join(model_dir, 'pytorch_model.bin'),
                config=config,
            )

            # classificationに使うdense layer以外のパラメータが更新されないようにfreeze
            if self.training_config.freeze_bert_params:
                logger.info('Freeze re-training in BERT Model.')
                for name, param in model.bert.named_parameters():
                    if not (name.startswith('encoder.layer.7') or name.startswith('pooler.')):
                        print(f'param.name={name}')
                        param.requires_grad = False

            logger.info(f"{model=}")

        except ValueError as err:
            sys.stderr.write('Error: モデルの読み込みに失敗しました。')
            sys.stderr.write(err)

        return model

    def _load_dataset(self) -> Dataset:
        """
        文書分類のための教師データをロードしてlabelごとに整理して返す

        Parameters
        ----------
        labels: List[str]
            分類するカテゴリーのラベル一覧

        Returns
        -------
            文章と対応するlabelをまとめたdict
        """
        dataset = {
            self._param_sentence_key: [],
            self._param_label_key: [],
        }
        max_load_num = int(self.training_config.train_data_max_per_label)

        # TODO 処理改善
        for label, category in enumerate(tqdm(self.categories)):
            print(f'{label=}, {category=}')
            for file in glob.glob(f'{self.training_config.train_dir}/{category}*'):
                print(f'file loading ....: {file}')
                load_count = 0
                with open(file) as f:
                    for i, sentence in enumerate(f, 2):
                        if random.randrange(i) % 7:  # ランダムな行を読み出す。
                            continue

                        dataset[self._param_sentence_key].append(sentence)
                        dataset[self._param_label_key].append(label)
                        load_count += 1
                        if load_count == max_load_num:
                            print(f'{category} dataset touch {max_load_num}. stop loading.')
                            break
            print(f'{category} dataset total {load_count} users.')

        return Dataset.from_dict(dataset)

    def compute_f1score(self, eval_pred):
        """
        Trainerのevaluation stepでf1 scoreを評価値として算出するための関数

        Parameters
        ----------
        eval_pred: EvalPredictions
            Trainerから渡された推測値とラベルのデータ

        Returns
        -------
            f1 score

        """
        _preds, labels = eval_pred
        argpreds = np.argmax(_preds, axis=1)
        eval = self.metrics.compute(predictions=argpreds, references=labels, average="macro")
        eval.update(load_metric('glue', 'sst2').compute(predictions=argpreds, references=labels))

        return eval

    def setup_training(self,
                       training_args: TrainingArguments,
                       training_rate: float = 0.8,
                       inspection_rate: float = 0.1
                       ) -> None:
        """
        トレーニングの前準備を行う

        Parameters
        ----------
        training_args: TrainingArguments
            トレーニングの設定
        training_rate: float
            トレーニングに使用するデータの比率
        inspection_rate: float
            検証に使用するデータの比率

        Returns
        -------
            None
        """
        def _tokenize(examples: dict) -> BertTokenizer:
            # データセットをトークン化する際に使用するインナー関数
            # Tokenize the texts
            return self.tokenizer(
                examples[self._param_sentence_key],
                padding="max_length",
                max_length=self.training_config.max_seq_length,
                truncation=True
            )

        # データセットの生成
        raw_dataset = self._load_dataset()
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_dataset = raw_dataset.map(
                _tokenize,
                batched=True,
                load_from_cache_file=not self.training_config.overwrite_cache,
                num_proc=self.training_config.num_proc,
                desc="Running tokenizer on dataset",
            )

        # 学習データ、検証データ、テストデータに分割
        n = len(raw_dataset)
        n_train = int(training_rate * n)
        n_val = int(inspection_rate * n)
        n_test = n_train + n_val
        raw_dataset = raw_dataset.shuffle(seed=42)
        # 学習データ
        dataset_train = raw_dataset[:n_train]
        # 検証データ
        dataset_val = raw_dataset[n_train:n_test]
        # テストデータ
        self.dataset_test = raw_dataset[n_test:]

        # Trainerの初期化
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=Dataset.from_dict(dataset_train),
            eval_dataset=Dataset.from_dict(dataset_val),
            tokenizer=self.tokenizer,
            data_collator=None,
            compute_metrics=self.compute_f1score
        )

    def start_training(self, save_dir: str) -> None:
        """
        モデルのトレーニングを行う

        Parameters
        ----------
        save_dir: str
            モデルを保存するディレクトリ

        Returns
        -------
        None
        """
        # checkpointフォルダがあるかチェック。存在してない場合はcheckpointを使わないようにする
        checkpoint_flg = glob.glob(f'{save_dir}/checkpoint*') != []
        if self.training_config.use_checkpoint and not checkpoint_flg:
            logging.warning(
                "Set use_checkpoint=True, but checkpoint dir is ont found.")
        # 学習の開始
        self.trainer.train(resume_from_checkpoint=(
            checkpoint_flg and self.training_config.use_checkpoint))
        # 学習終了時のモデルを保存する
        self.trainer.save_model(save_dir)

    def model_score(self) -> None:
        """
        トレーニングしたモデルの評価を行う

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # モデルの精度をF1 Scoreで計測
        predictions = self.trainer.predict(test_dataset=Dataset.from_dict(self.dataset_test))
        preds = np.argmax(predictions.predictions, axis=1)
        metrics = load_metric('f1')
        f1_score = metrics.compute(
            predictions=preds, references=predictions.label_ids, average="macro")

        print(f'{preds=}')
        print(f'{predictions.label_ids=}')
        print(f1_score)


def main():
    """
    番組の視聴予測をfine-tuningするプログラム

    視聴データで事前学習済みのBERTモデルに対して、文章分類タスクをfine-tuningする。
    教師データとして、ある番組を視聴した/していないデータを与えて学習させる。
    結果として、事前の視聴行動からある番組を学習したかしていないかを予測するモデルが作成される。

    必須の引数として、事前学習済みのBERTモデルのフォルダ、教師データのフォルダ、学習結果を保存するフォルダの3つを指定する必要がある。
    その他、BERTへの入力単語数(max_seq_length)やTrainingArgumentsも引数として指定可能

    TrainingArgumentsの引数についてはこちらを参照
    https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/trainer#transformers.TrainingArguments

    Usage
    -----
        (set with training options)
        python3 bert_cls.py --train_dir ./data/classifys --model_dir ./pretraining_data/models \
        --output_dir classify_task --num_proc=4 --max_seq_length 128 --do_train --eval_step 100 \
        --evaluation_strategy steps --train_data_max_per_label 300 --num_train_epochs=10 \
        --logging_strategy steps --logging_step 20  --load_best_model_at_end True

    Examples
    --------
        python3 bert_cls.py --train_dir ./data/classifys --model_dir ./pretraining_data/models \
        --output_dir classify_task
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser((ClassifyTrainingConfg, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()

    # ログレベルを設定
    logging.basicConfig(level=training_args.log_level)
    # 分類カテゴリ名リスト
    category_labels = ['viewed', 'not_viewed']
    # 分類問題タスクのmodelを保存するディレクトリ
    classify_model_dir = os.path.join(training_args.output_dir, 'model')
    # random seedを初期化
    random.seed(training_args.seed)

    # トレーニング開始
    classifyst_training = ClassifyTraining(data_args, category_labels, data_args.model_dir)
    classifyst_training.setup_training(training_args)
    classifyst_training.start_training(classify_model_dir)

    # モデルの評価
    classifyst_training.model_score()


if __name__ == '__main__':
    main()
