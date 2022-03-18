import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tokenizer import ByT5KoreanTokenizer

# Huggingface ByT5 tokenizer
#   0 <pad>
#   1 <unk>
#   2 <eos>
#   3~258 utf-8 encoding
#   259~384 <extra_id_0>~<extra_id_125>
#
# note. ByT5 논문에서는 extra_ids가 258부터 감소하는 방식이지만, Huggingface에서는 259부터 증가하는 방식으로 구현되어 있음
#       huggingface tokenizer는 <extra_id_0>이 259, 학습된 구글 모델에서는 258
tokenizer_hf = AutoTokenizer.from_pretrained('google/byt5-small')
# ByT5_Korean tokenizer
#   0 <pad>
#   1 <unk>
#   2 <eos>
#   3~258 utf-8 encoding
#   259~277 초성ㄱ~ㅎ
#   279~299 중성ㅏ~ㅣ
#   300~327 무종성,종성ㄱ~종성ㅎ
#   328~384 <extra_id_0>~<extra_id_56>
tokenizer_jamo = ByT5KoreanTokenizer()

model_google = T5ForConditionalGeneration.from_pretrained('google/byt5-large')
# model_jamo = T5ForConditionalGeneration.from_pretrained('/data/youngjin/projects/byt5-Korean/models/byt5-korean-ko7en3-large-128bs-adafactor-1e-2linear-100000steps-ds-0214-1201/checkpoint-100000')
model_jamo = T5ForConditionalGeneration.from_pretrained('everdoubling/byt5-Korean-large')

# input_ids_hf = tokenizer_hf("에버더블링은 서울에 위치한 <extra_id_0>으로 한국어를 자모 단위로 학습하는 언어 모델을 <extra_id_1>하였습니다.").input_ids
input_sentence = '한국어 위키백과(영어: Korean Wikipedia)는 한국어로 운영되는 위키백과의 다언어판 가운데 하나로서, 2002년 10월 11일에 <extra_id_0>. 또한 현재 한국어 위키백과에는 넘겨주기, 토론, 그림 등 페이지로 불리는 모든 문서를 포함하면 총 2,629,860개가 <extra_id_1>되어 있으며, 넘겨주기를 포함한 일반 문서 수는 1,278,560개,[1] 그중 넘겨주기, 막다른 문서를 제외한 일반 문서 수는 573,149개이다.'

input_ids_hf = tokenizer_hf(input_sentence).input_ids
input_ids_google = [x if x<=258 else 258-(x-259) for x in input_ids_hf]
outputs_google = model_google.generate(torch.tensor([input_ids_google]))
# print(bytes(outputs_google[0][2:] - 3).decode('utf-8', errors='ignore'))
outputs_hf = [x if x<=258-10 else 259+(258-x) for x in outputs_google[0].tolist()]
print(tokenizer_hf.decode(outputs_hf))

input_ids_jamo = tokenizer_jamo(input_sentence).input_ids
outputs_jamo = model_jamo.generate(torch.tensor([input_ids_jamo]))
print(tokenizer_jamo.decode(outputs_jamo[0]))

a = 0
# input_ids = list("한국어 위키백과(영어: Korean Wikipedia)는 한국어로 운영되는 위키백과의 다언어판 가운데 하나로서, 2002년 10월 11일에 ".encode("utf-8")) + list(b'\xff') + list(". 또한 현재 한국어 위키백과에는 넘겨주기, 토론, 그림 등 페이지로 불리는 모든 문서를 포함하면 총 2,629,860개가 수록되어 있으며, 넘겨주기를 포함한 일반 문서 수는 1,278,560개,[1] 그중 넘겨주기, 막다른 문서를 제외한 일반 문서 수는 573,149개이다. ".encode("utf-8"))
# input_ids.append(-2) # end of sequence = 1
# input_ids = torch.tensor([input_ids]) + 3
# # input_ids = tensor([[239, 153, 148, 239, 170, 135, 240, 155, 187, 238, 141, 151,  35, 240,
# #                      152, 159, 237, 184, 176, 239, 160, 155,  35, 238, 146, 135, 239, 142,
# #                      159,  35, 239, 135, 159, 239, 157, 187, 239, 154, 147, 239, 135, 159,
# #                      35, 258, 239, 161, 136, 238, 142, 139, 238, 142, 167,  49,   1]])
