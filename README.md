# byt5-Korean
한국어에 특성에 맞게 ByT5를 개선한 언어 모델을 제안합니다. (2022-01-28 개발/학습 진행 중)
한국어의 특성상 mT5대비 ByT5가 여러 태스크에서 좋은 성능을 보여주지만, ByT5가 채택하고 있는 utf-8인코딩 역시 한국어에 최적화 된 형태는 아닙니다. 여기에서는 자모 단위의 한국어 특화 인코딩을 추가한 ByT5-Korean 모델을 학습(pretrain)하는 코드를 제공합니다.
Google이 공개한 T5(+mesh_tensorflow) 기반의 코드를 이용해서 학습하는 코드와, Huggingface transformers(+deepspeed) 라이브러리를 이용한 학습 코드를 모두 제공합니다.

## Byte encoding for Korean

## Dataset
학습에는 [mc4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual_nights_stay) 한국어, 영어 말뭉치가 사용되었습니다. 설치 방법은 [T5 github](https://github.com/google-research/text-to-text-transfer-transformer)에 공개되어 있습니다만, GCS bucket 기반으로  Google Cloud Dataflow를 이용하여 전처리를 해야 하며 비용이 100만원 이상 드는 것으로 알려져 있습니다. 다행이 전처리가 완료된 데이터셋을 다운로드 하는 방법을 [여기](https://github.com/allenai/allennlp/discussions/5056)에서 설명하고 있습니다. TFDS와 JSON 두 가지 포맷이 있는데, TFDS는 requester-pays bucket에서 받아야 하기 때문에 몇십~몇백 달러 정도의 비용이 청구될 수 있습니다. 본 repo에서는 두 가지 포맷 모두 지원하고 있지만, 원 논문과 일치하는 데이터셋 구성을 원한다면 TFDS 데이터를 다운받기를 권장드립니다. 

### mC4 한국어 TFDS 받기
```sh
mkdir -p local_datasets_dir/c4/multilingual/3.0.1/
gsutil -m cp 'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/c4-ko*' local_datasets_dir/c4/multilingual/3.0.1/
```

## Pretraining

### Mesh_tensorflow

### Huggingface transformers
```sh
python pretrainer.py train_config.json
```

### Huggingface transformers + Deepspeed
```sh
deepspeed pretrainer.py train_config_deepspeed.json
```

### Requirements

## Fine-tuning

## References

[t5] https://github.com/google-research/text-to-text-transfer-transformer
[mt5] https://github.com/google-research/multilingual-t5
[byt5] https://github.com/google-research/byt5
