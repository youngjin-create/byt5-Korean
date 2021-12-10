from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained('google/byt5-large')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-large')

input_ids = list("양진호는 한국의 도시 서울에서 ".encode("utf-8")) + list(b'\xff') + list("입니다.".encode("utf-8"))
input_ids.append(-2) # end of sequence = 1
input_ids = torch.tensor([input_ids]) + 3
# input_ids = tensor([[239, 153, 148, 239, 170, 135, 240, 155, 187, 238, 141, 151,  35, 240,
#                      152, 159, 237, 184, 176, 239, 160, 155,  35, 238, 146, 135, 239, 142,
#                      159,  35, 239, 135, 159, 239, 157, 187, 239, 154, 147, 239, 135, 159,
#                      35, 258, 239, 161, 136, 238, 142, 139, 238, 142, 167,  49,   1]])

outputs = model.generate(input_ids)
print(bytes(outputs[0][2:] - 3).decode('utf-8')) # 살 수 있는 양
