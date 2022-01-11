
import torch

from transformers import T5Config, T5Model, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from tokenizer import ByT5KoreanTokenizer
import dataset

# import datetime
# model_path = './byT5-Korean-small-{}'.format(datetime.datetime.now())
model_path = './byT5-Korean-small-5'

model_config_small = {
    "_name_or_path": "google/byt5-small",
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 3584,
    "d_kv": 64,
    "d_model": 1472,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "gradient_checkpointing": False,
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 4,
    "num_heads": 6,
    "num_layers": 12,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "tokenizer_class": "ByT5Tokenizer",
    "transformers_version": "4.12.5",
    "use_cache": True,
    "vocab_size": 384
}

model_config_large = {
    "_name_or_path": "google/byt5-large",
    "architectures": [
        "T5ForConditionalGeneration"
    ],
    "d_ff": 3840,
    "d_kv": 64,
    "d_model": 1536,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "gated-gelu",
    "gradient_checkpointing": False,
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "num_decoder_layers": 12,
    "num_heads": 16,
    "num_layers": 36,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": False,
    "tokenizer_class": "ByT5Tokenizer",
    "transformers_version": "4.12.5",
    "use_cache": True,
    "vocab_size": 384
}

def train():
    model_config = T5Config(**model_config_small)
    # model = T5Model(config=model_config) # 그냥 T5 모델로 하면 안되고 T5ForConditionalGeneration를 사용해야 함
    model = T5ForConditionalGeneration(config=model_config)

    training_args = TrainingArguments(
        output_dir=model_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=1,             # number of training epochs, feel free to tweak
        per_device_train_batch_size=8,  # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=4,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=8,   # evaluation batch size
        logging_steps=1000,              # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        load_best_model_at_end=True,    # whether to load the best model (in terms of loss) at the end of training
        save_total_limit=3,             # whether you don't have much space so you let only 3 model weights saved in the disk
        ignore_data_skip=True,
        # dataloader_num_workers=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=AutoTokenizer.from_pretrained('google/byt5-small'), model='google/byt5-small'),
        data_collator=DataCollatorForSeq2Seq(tokenizer=ByT5KoreanTokenizer()),
        train_dataset=dataset.KoreanDataset(evaluate=False),
        eval_dataset=dataset.KoreanDataset(evaluate=True),
    )

    trainer.train()
    # trainer.train('byT5-Korean-small-2/checkpoint-4500') # continue training

def test_base():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
def test():
    tokenizer = ByT5KoreanTokenizer()
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained(model_path + '/checkpoint-103000')
    ds_train = dataset.KoreanDataset(evaluate=False)
    for i in range(10):
        # input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        # labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
        input_ids = torch.unsqueeze(ds_train[i]['input_ids'], 0).type(torch.long)
        labels = torch.unsqueeze(ds_train[i]['labels'], 0).type(torch.long)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        output_ids = model.generate(input_ids)[0][1:] # ignore output_ids[0] which is decoder_start_token_id
        print(tokenizer.decode(input_ids[0]))
        print(tokenizer.decode(output_ids))

if __name__ == "__main__":
    # train()
    test()
    print("Done")
