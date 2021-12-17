
from transformers import T5Config, T5Model, TrainingArguments, Trainer
import mydataset

model_path = './byT5-Korean-large'

model_config_dict = {
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

model_config = T5Config(**model_config_dict)
model = T5Model(config=model_config)

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=500,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=500,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)

trainer = Trainer(
    model=model,
    args=training_args,
    # data_collator=data_collator,
    train_dataset=mydataset.KoreanDataset(evaluate=False),
    eval_dataset=mydataset.KoreanDataset(evaluate=True),
)

trainer.train()

a = 0
