# byt5-Korean
한국어에 특성에 맞게 ByT5를 개선한 언어 모델을 제안합니다. (2022-01-16 개발/학습 진행 중)

한국어의 특성상 mT5대비 ByT5가 여러 태스크에서 좋은 성능을 보여주지만, ByT5가 채택하고 있는 utf-8인코딩 역시 한국어에 최적화 된 형태는 아닙니다. 여기에서는 자모 단위의 한국어 특화 인코딩을 추가한 ByT5-Korean 모델을 학습(pretrain)하는 코드를 제공합니다.

Google이 공개한 T5(+mesh_tensorflow) 기반의 코드를 이용해서 학습하는 코드와, Huggingface transformers(+deepspeed) 라이브러리를 이용한 학습 코드를 모두 제공합니다.

## Byte encoding for Korean

## Dataset
mc4 데이터셋을 다운로드하는 방법을 설명합니다.

## Pretraining

### Requirements

## Fine-tuning

## References

[t5] https://github.com/google-research/text-to-text-transfer-transformer
[mt5] https://github.com/google-research/multilingual-t5
[byt5] https://github.com/google-research/byt5
