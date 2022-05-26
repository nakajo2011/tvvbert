# -*- coding: utf-8 -*-
import glob
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (AutoModelForMaskedLM, BertConfig, BertTokenizerFast,
                          DataCollatorForLanguageModeling, HfArgumentParser,
                          Trainer, TrainingArguments)


logger = logging.getLogger(__name__)


@dataclass
class PreModelConfig:
    """
    事前学習するBERTモデルの各種パラメータの設定.
    """

    max_position_embeddings: int = field(
        default=512,
        metadata={"help": "The word length of a sequence(and BERT input)."}
    )

    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "The layer quantity of BERT Encoder"}
    )

    intermediate_size: int = field(
        default=3072,
        metadata={"help": "The vector size of Intermediate Layer"}
    )
    num_attention_heads: int = field(
        default=12,
        metadata={"help": "The quantity of multi-head attention."}
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "The vector size of a attention head."}
    )


@dataclass
class PreTrainingConfig:
    """
    事前学習データと学習済みmodelの出力先ディレクトリなどの設定.
    """

    vocab_file: str = field(
        metadata={"help": "The vocabulary list file for Tokenizer."}
    )
    train_file: str = field(
        metadata={"help": "The input training data file (a text file)."},
    )
    use_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "checkpointから学習を再開するかどうか。default True"
            "指定されたoutput_dirの中にあるcheckpointを利用する。checkpointが存在しない場合はこのフラグは無視される"
        },
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
    eval_percentage: int = field(
        default=5,
        metadata={"help": "検証データに割り当てる件数。％で指定する。"}
    )


class BertPreTraining(object):
    """
    BERTの事前学習を行うクラス

    Attributes
    ----------
    model: BertForSequenceClassification
        モデル
    tokenizer: BertTokenizer
        モデルに対応するトークナイザ
    trainer: Trainer
        トレーナー
    dataset_test: Dataset
        検証時に使用するデータセット
    """

    def __init__(self, model_config: PreModelConfig, tokenizer_dir: str) -> None:
        """
        Parameters
        ----------
        model_config : PreModelConfig
            生成するモデルの設定情報
        tokenizer_dir : str
            トークナイザのディレクトリパス
        """
        self.tokenizer = self._generate_tokenizer(tokenizer_dir)
        self.model = self._generate_model(model_config)
        self.trainer = None

        # GPUを使わない場合
        if not torch.cuda.is_available():
            logger.warning("GPU is not available. Using CPU.")
            self.model.to('cpu')

    def _generate_tokenizer(self, tokenizer_dir: str) -> BertTokenizerFast:
        """
        対象のディレクトリにトークナイザを生成する

        Parameters
        ----------
        tokenizer_dir : str
            トークナイザのディレクトリパス

        Returns
        -------
        tokenizer : BertTokenizerFast
            トークナイザ
        """
        return BertTokenizerFast.from_pretrained(tokenizer_dir)

    def _generate_model(self, model_config: PreModelConfig) -> AutoModelForMaskedLM:
        """
        BERTモデルを生成する

        Parameters
        ----------
        model_config : PreModelConfig
            生成するモデルのコンフィグ

        Returns
        -------
        model : AutoModelForMaskedLM
            生成したBERTモデル
        """
        vocab_size = len(self.tokenizer)
        # BERTコンフィグ設定
        config = BertConfig(**asdict(model_config))
        model = AutoModelForMaskedLM.from_config(config)
        model.resize_token_embeddings(vocab_size)

        logger.info(f"{model=}")

        return model

    def setup_training(self,
                       data_files: Tuple,
                       training_config: PreTrainingConfig,
                       training_params: TrainingArguments,
                       max_word_length: int,
                       column_name: str = 'text',
                       data_type: str = 'text'
                       ) -> None:
        """
        トレーニングの前準備を行う

        Parameters
        ----------
        data_files: Tuple
            トレーニングデータ
        training_config: PreTrainingConfig
            トレーニングの設定情報
        training_params: TrainingArguments
            トレーニングのパラメータ
        max_word_length: int
            トークン生成時の最大単語数
        column_name: str
            カラム名
        data_type: str
            読み込むデータの種類

        Returns
        -------
            None
        """
        def _tokenize(examples):
            # データセットをトークン化する際に使用するインナー関数
            # Tokenize the texts

            # Remove empty lines
            examples[column_name] = [line for line in examples[column_name] if line.split()]

            return self.tokenizer(
                examples[column_name],
                padding="max_length",
                truncation=True,
                max_length=max_word_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        # データセット読み込み
        validation_data_rate = int(training_config.eval_percentage)
        raw_datasets = load_dataset(
            data_type,
            data_files=data_files
        )
        raw_datasets["validation"] = load_dataset(
            data_type,
            data_files=data_files,
            split=f"train[:{validation_data_rate}%]"
        )
        raw_datasets["train"] = load_dataset(
            data_type,
            data_files=data_files,
            split=f"train[{validation_data_rate}%:]"
        )

        with training_params.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                _tokenize,
                batched=True,
                num_proc=training_config.num_proc,
                remove_columns=[column_name],
                load_from_cache_file=not training_config.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )

        # 学習データを提供するcollatorを作成
        DATA_COLLATOR = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=None
        )

        # 学習器の生成
        self.trainer = Trainer(
            model=self.model,
            args=training_params,
            data_collator=DATA_COLLATOR,
            tokenizer=self.tokenizer,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
        )

    def start_training(self, save_dir: str, use_checkpoint) -> None:
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
        checkpoint_flg = (glob.glob(f'{save_dir}/checkpoint*') != [])
        if use_checkpoint and not checkpoint_flg:
            logger.warning("Set use_checkpoint=True, but checkpoint dir is ont found.")

        self.trainer.train(resume_from_checkpoint=checkpoint_flg and use_checkpoint)
        # 学習終了時のモデルを保存する
        self.trainer.save_model(save_dir)


def main():
    """
    BERT-pretraining
    # 参考 https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py

    視聴データを事前学習するプログラム
    必須の引数として、訓練用の文章ファイルのパス、単語一覧ファイルのパス、学習結果を保存するフォルダの3つを指定する必要がある。
    その他、BERTモデルのハイパーパラメータや、TrainingArgumentsもオプションとして指定可能。

    TrainingArgumentsの引数についてはこちらを参照：
    https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/trainer#transformers.TrainingArguments

    Usage:
    -----
    python3 bert_pretraining.py \
    --train_file data/sentence.txt --vocab_file data/vocab.txt --output_dir pretraining_data/

    モデルのパラメータや学習条件を指定することも可能:
    python3 bert_pretraining.py \
    --train_file data/sentence.txt --vocab_file data/vocab.txt --output_dir pretraining_data/ \
    --max_position_embeddings=128 --num_hidden_layers=4 --num_attention_heads=4 --hidden_size=128 \
    --per_device_train_batch_size=2 --num_train_epochs=1 --save_total_limit=2

    ※異なるパラメータモデルではcheckpointからの再学習は行えいないことに注意。
    """
    # コマンドライン引数を取得
    parser = HfArgumentParser((PreModelConfig, PreTrainingConfig, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ログレベルを設定
    logging.basicConfig(level=training_args.log_level)
    # 事前学習に使用する文章データのファイル
    data_files = (data_args.train_file,)
    # 学習モデルを保存するディレクトリ
    model_dir = os.path.join(training_args.output_dir, 'models')
    # トークナイザを保存するディレクトリ
    tokenizer_dir = model_dir

    # モデルを保存するフォルダを作成して。vocabファイルをコピー
    os.makedirs(model_dir, exist_ok=True)
    shutil.copyfile(data_args.vocab_file, os.path.join(model_dir, "vocab.txt"))

    # 学習の開始
    bert_pre_training = BertPreTraining(model_args, tokenizer_dir)
    bert_pre_training.setup_training(data_files,
                                     data_args,
                                     training_args,
                                     model_args.max_position_embeddings,
                                     )
    bert_pre_training.start_training(model_dir, data_args.use_checkpoint)

    print("pre-training completed!!!!!!!!!")


if __name__ == '__main__':
    main()
