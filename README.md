# byt5-Korean
한국어에 특성에 맞게 ByT5를 개선한 언어 모델을 제안합니다. 
한국어의 특성상 mT5대비 ByT5가 여러 태스크에서 좋은 성능을 보여주지만, ByT5가 채택하고 있는 utf-8인코딩 역시 한국어에 최적화 된 형태는 아닙니다. 여기에서는 자모 단위의 한국어 특화 인코딩을 추가한 ByT5-Korean 모델을 학습(pretrain)하는 코드를 제공합니다.
Google이 공개한 T5(+mesh_tensorflow) 기반의 코드를 이용해서 학습하는 코드와, Huggingface transformers(+deepspeed) 라이브러리를 이용한 학습 코드를 모두 제공합니다.

## Updates
2022-03-04 Huggingface에 large 모델 공개

## Byte encoding for Korean

ByT5Korean 모델은 한국어 자모별로 하나의 토큰을 할당합니다.

```text
id: token
0: <pad>
1: <eos>
2: <unk>
3~258: utf-8 encoding
259~277: beginning consonants(초성), 19개(ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ)
278~298: middle vowel(중성), 21개(ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ)
299~326: final consonant(종성), 무종성+27개(ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ)
327~384: from <extra_id_0> to <extra_id_57>
```

ByT5KoreanTokenizer.py 파일에 토크나이저가 구현되어 있습니다. 실행 예는 다음과 같습니다.
```python
tokenizer_jamo = ByT5KoreanTokenizer()
print(tokenizer_jamo('가힣abc 안녕하세요')['input_ids'])
# [259, 278, 299, 277, 298, 326, 100, 101, 102, 35, 270, 278, 303, 261, 284, 320, 277, 278, 299, 268, 283, 299, 270, 290, 299, 1]
```

## Dataset
학습에는 [mc4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual_nights_stay) 한국어, 영어 말뭉치가 사용되었습니다. 설치 방법은 [T5 github](https://github.com/google-research/text-to-text-transfer-transformer)에 공개되어 있습니다만, GCS bucket 기반으로  Google Cloud Dataflow를 이용하여 전처리를 해야 하며 비용이 100만원 이상 드는 것으로 알려져 있습니다. 다행이 전처리가 완료된 데이터셋을 다운로드 하는 방법을 [여기](https://github.com/allenai/allennlp/discussions/5056)에서 설명하고 있습니다. TFDS와 JSON 두 가지 포맷이 있는데, TFDS는 requester-pays bucket에서 받아야 하기 때문에 몇십~몇백 달러 정도의 비용이 청구될 수 있습니다. 본 repo에서는 두 가지 포맷 모두 지원하고 있지만, 원 논문과 일치하는 데이터셋 구성을 원한다면 TFDS 데이터를 다운받기를 권장드립니다. 

### mC4 한국어 TFDS 받기
```sh
gcloud auth login
gcloud config set project 'my-google-project-name'
mkdir -p local_datasets_dir/c4/multilingual/3.0.1/
gsutil -m -u 'my-google-project-name' cp 'gs://allennlp-tensorflow-datasets/c4/multilingual/3.0.1/c4-ko*' local_datasets_dir/c4/multilingual/3.0.1/
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

### Pretrained model inference example
```python
import torch
from tokenizer import ByT5KoreanTokenizer # https://github.com/everdoubling/byt5-Korean
from transformers import T5ForConditionalGeneration

tokenizer_jamo = ByT5KoreanTokenizer()
model = T5ForConditionalGeneration.from_pretrained('everdoubling/byt5-Korean-large')

input_sentence = '한국어 위키백과(영어: Korean Wikipedia)는 한국어로 운영되는 위키백과의 다언어판 가운데 하나로서, 2002년 10월 11일에 <extra_id_0>. 또한 현재 한국어 위키백과에는 넘겨주기, 토론, 그림 등 페이지로 불리는 모든 문서를 포함하면 총 2,629,860개가 <extra_id_1>되어 있으며, 넘겨주기를 포함한 일반 문서 수는 1,278,560개,[1] 그중 넘겨주기, 막다른 문서를 제외한 일반 문서 수는 573,149개이다.'

input_ids_jamo = tokenizer_jamo(input_sentence).input_ids
outputs_jamo = model_jamo.generate(torch.tensor([input_ids_jamo]))
print(tokenizer_jamo.decode(outputs_jamo[0]))
# <pad><extra_id_0>설립되었다<extra_id_1>đě
```

## Fine-tuning


## References

[t5] https://github.com/google-research/text-to-text-transfer-transformer
[mt5] https://github.com/google-research/multilingual-t5
[byt5] https://github.com/google-research/byt5
