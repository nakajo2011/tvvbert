# README #
* 「Explainability of Transformer for Large Over-the-Top Media Viewing Logs」に用いたBERTのpre-trainingとfine-tuningの実装
* Attention Flow/Rolloutの計算とvisualizationの実装

# How do I get set up?(requirement)
* pytorch
* huggingface/transformers
* huggingface/datasets

see 'requirements.txt' for more dependencies.

# How to use

## pre-training
python3 bert_pretraining.py pretraining_config.json

see 'bert_pretraining.py' for more details.

## fine-tuning
python3 bert_cls.py cls_config.json

see 'bert_cls.py' for more details.

# Visualizing
参考にしたリポジトリ： https://github.com/samiraabnar/attention_flow

## How to use
```python3 attention_visualization_v4.py```
