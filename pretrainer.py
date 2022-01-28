
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# os.environ["WANDB_DISABLED"] = "true"

sys.path = [p for p in sys.path if 'home' not in p]

import torch

import transformers
from transformers import (
    HfArgumentParser,
    T5Config, AutoConfig,
    T5ForConditionalGeneration,
    T5Tokenizer, AutoTokenizer,
    DataCollatorForSeq2Seq,
    Adafactor,
    Seq2SeqTrainingArguments, TrainingArguments,
    Seq2SeqTrainer, Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from tokenizer import ByT5KoreanTokenizer
import dataset
import dataset_torch

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pretrain_config_name: Optional[str] = field(
        default=None, metadata={"help": "Config name for pretraining"}
    )
    pretrain_from_scratch: bool = field(default=True, metadata={"help": "Whether to pretrain from scratch"})
    tokenizer_name: Optional[str] = field(
        default="utf8-google", metadata={"help": "Tokenizer name", "choices": ["utf8-google", "utf8-extra", "utf8-korean"]}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    if sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args, training_args = parser.parse_args_into_dataclasses(sys.argv[1:])# + ['--output_dir', model_path])

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    print(model_args.tokenizer_name)
    if model_args.tokenizer_name == "utf8-google":
        train_dataset = dataset_torch.MyIterableDataset(mixture_or_task_name='google.ko') # sentinel_ids = [258, 257, 256, ...], original byt5 encoding
    elif model_args.tokenizer_name == "utf8-extra":
        train_dataset = dataset_torch.MyIterableDataset(mixture_or_task_name='extra.ko') # sentinel_ids = [259, 260, 261, ...], modified encoding
    else:
        train_dataset = dataset_torch.MyIterableDataset(mixture_or_task_name='byt5_korean.ko')
    # train_dataset = dataset.KoreanDataset(evaluate=False)
    eval_dataset = dataset.KoreanDataset(evaluate=True)

    tokenizer = ByT5KoreanTokenizer()
    if model_args.pretrain_from_scratch:
        config = AutoConfig.from_pretrained(model_args.pretrain_config_name)
        config.dropout_rate = 0.0
        model = T5ForConditionalGeneration(config=config)
    else:
        # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)#, config=config)
        # model = T5ForConditionalGeneration.from_pretrained('/data/youngjin/projects/t5-family/myt5/modelhf/pytorch_model')

    # model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Configure optimizer
    custom_optimizer = None
    if training_args.deepspeed and training_args.adafactor: # and not training_args.sharded_ddp:
        optimizer_grouped_parameters = []
        for _, p in model.named_parameters():
            optimizer_grouped_parameters.append({
                "params": [p],
                "weight_decay": 0.0,
            })
        optimizer_kwargs = {"lr": training_args.learning_rate, "scale_parameter": True, "relative_step": False}
        custom_optimizer = Adafactor(optimizer_grouped_parameters, **optimizer_kwargs)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(custom_optimizer, None)
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = training_args.generation_max_length if training_args.generation_max_length is not None else 256
    num_beams = training_args.generation_num_beams if training_args.generation_num_beams is not None else 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams)
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(eval_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                import re
                predictions = [re.sub('(<pad>)*$', '', pred) for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results

if __name__ == "__main__":
    main()
