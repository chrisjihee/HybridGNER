import logging
import os
import sys
from typing import Callable, Optional, List, Dict
from unittest.mock import patch

import datasets
import numpy as np
import torch
import typer
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import gather_object
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from datasets.utils.logging import set_verbosity as datasets_set_verbosity
from typing_extensions import Annotated

import gner
import transformers
import transformers.utils.logging
from chrisbase.data import AppTyper, JobTimer, Counter, NewProjectEnv
from chrisbase.io import LoggingFormat, LoggerWriter, set_verbosity_info, set_verbosity_debug, new_path, convert_all_events_in_dir, set_verbosity_warning
from chrisbase.time import from_timestamp, now_stamp
from chrisdata.ner import GenNERSampleWrapper
from gner.arguments import TrainingArgumentsForAccelerator, CustomDataArguments, ExSeq2SeqTrainingArguments
from progiter import ProgIter
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerBase,
    BatchEncoding,
    PrinterCallback,
    set_seed,
)
from transformers.trainer_utils import TrainOutput, PredictionOutput
from transformers.utils import is_torch_tf32_available, is_torch_bf16_gpu_available
from transformers.utils.logging import set_verbosity as transformers_set_verbosity

# Global settings
logger: logging.Logger = logging.getLogger("gner")


def update_progress(
        counter: Counter,
        rank: int = -1,
        pbar: Optional[ProgIter] = None,
):
    count = counter.inc()
    if pbar and rank == 0:
        pbar.step(min(count - pbar._iter_idx, pbar.total - pbar._iter_idx), force=count >= pbar.total)


def preprocess_row(
        row: LazyRow,
        rank: int,
        is_encoder_decoder: bool,
        max_source_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizerBase,
        counter: Counter,
        update: Optional[Callable[[Counter, int], None]] = None,
) -> BatchEncoding:
    example = GenNERSampleWrapper.model_validate(row)

    def tokenize_encoder_decoder_train():
        model_inputs = tokenizer(
            text=example.instance.instruction_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        model_inputs["labels"] = tokenizer(
            text_target=example.instance.prompt_labels,
            max_length=max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )['input_ids']
        return model_inputs

    def tokenize_encoder_decoder_infer():
        model_inputs = tokenizer(
            text=example.instance.instruction_inputs,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        return model_inputs

    def tokenize_decoder_only_train():
        prompt_input = f"[INST] {example.instance.instruction_inputs} [/INST]"
        model_inputs = tokenizer(
            text=f"{prompt_input} {example.instance.prompt_labels}",
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
            model_inputs["input_ids"].append(tokenizer.eos_token_id)
            model_inputs["attention_mask"].append(1)
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        prompt_tokens = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )["input_ids"]

        if prompt_tokens[-1] == tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]

        if len(prompt_tokens) > len(model_inputs["labels"]):
            raise ValueError(
                f"Prompt is longer than the input, something went wrong. Prompt: {prompt_tokens}, input:"
                f" {model_inputs['input_ids']}"
            )

        for i in range(len(prompt_tokens)):
            model_inputs["labels"][i] = -100

        return model_inputs

    def tokenize_decoder_only_infer():
        prompt_input = f"[INST] {example.instance.instruction_inputs} [/INST]"
        model_inputs = tokenizer(
            text=prompt_input,
            max_length=max_source_length + max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=True,
        )
        if model_inputs["input_ids"][-1] == tokenizer.eos_token_id:
            model_inputs["input_ids"].pop()
            model_inputs["attention_mask"].pop()
        return model_inputs

    # Check if this row belongs to the training split
    is_train = (example.split == "train")

    if is_encoder_decoder:
        if is_train:
            tokenized_sample = tokenize_encoder_decoder_train()
        else:
            tokenized_sample = tokenize_encoder_decoder_infer()
    else:
        if is_train:
            tokenized_sample = tokenize_decoder_only_train()
        else:
            tokenized_sample = tokenize_decoder_only_infer()

    # Update progress if a callback is provided
    if update:
        update(counter=counter, rank=rank)

    tokenized_sample["time_stamp"] = now_stamp()
    tokenized_sample["tokenizer_name"] = tokenizer.name_or_path
    return tokenized_sample


def preprocess_dataset(
        split: datasets.Split,
        raw_datasets: datasets.DatasetDict,
        is_encoder_decoder: bool,
        max_source_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizerBase,
        local_rank: int,
        max_workers: int,
        progress_seconds: float = 2.0,
        cache_file: Optional[str] = None,
) -> Optional[datasets.Dataset]:
    if split not in raw_datasets or len(raw_datasets[split]) == 0:
        return None

    # Prepare a progress bar
    with ProgIter(
            time_thresh=progress_seconds,
            verbose=3,
            stream=LoggerWriter(logger),
            total=len(raw_datasets[split]),
            desc=f"Preprocess {split} dataset:"
    ) as pbar:
        # Disable the default progress bar in datasets
        datasets.disable_progress_bar()

        # Map function over the dataset
        counter = Counter(step=max_workers)
        dataset = raw_datasets[split].map(
            batched=False,
            function=preprocess_row,
            with_rank=True,
            fn_kwargs={
                "is_encoder_decoder": is_encoder_decoder,
                "max_source_length": max_source_length,
                "max_target_length": max_target_length,
                "tokenizer": tokenizer,
                "counter": counter,
                "update": lambda *vs, **ws: update_progress(
                    *vs, **ws, pbar=pbar if local_rank == 0 else None
                ),
            },
            load_from_cache_file=cache_file is not None,
            cache_file_name=cache_file,
            num_proc=max_workers,
        )

        # Re-enable datasets progress bars
        datasets.enable_progress_bars()

    # Log the last timestamp recorded (if any) in the dataset
    if len(dataset) > 0 and "time_stamp" in dataset.column_names and "tokenizer_name" in dataset.column_names:
        timestamp_str = from_timestamp(max(dataset["time_stamp"]))
        tokenizer_name = ', '.join(set(dataset["tokenizer_name"]))
        logger.info(f"Completed preprocessing for {split} dataset by {tokenizer_name} at {timestamp_str}")

    return dataset


def eval_predictions(dataset, preds, tokenizer, is_encoder_decoder, output_dir=None, save_prefix=None, save_suffix=None, write_predictions=False, accelerator=None):
    num_samples = len(dataset)
    assert num_samples > 0, f"eval dataset is empty"
    assert num_samples == len(preds), f"dataset(={num_samples}) != preds(={len(preds)})"
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if not is_encoder_decoder:
        match_pattern = "[/INST]"
        for i, preds in enumerate(decoded_preds):
            decoded_preds[i] = preds[preds.find(match_pattern) + len(match_pattern):].strip()

    all_examples: List[GenNERSampleWrapper] = [GenNERSampleWrapper.model_validate(example) for example in dataset]
    for idx, decoded_pred in enumerate(decoded_preds):
        all_examples[idx].instance.prediction_output = decoded_pred

    results = gner.compute_metrics2(all_examples, tokenizer=tokenizer, detailed=False, average_key="average")
    if write_predictions and output_dir is not None and save_prefix is not None:
        suffix = f"_{save_suffix}" if save_suffix else ""
        file_name = f"{save_prefix}-text_generations{suffix}.jsonl"
        with open(os.path.join(output_dir, file_name), "w") as fout:
            for example in all_examples:
                fout.write(example.model_dump_json() + "\n")
    return results


# Reference for implementation
# [1]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# [3]: https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [4]: https://huggingface.co/docs/transformers/en/main_classes/trainer
# [5]: https://huggingface.co/docs/transformers/en/main_classes/logging
def main(
        # for CustomDataArguments
        pretrained: Annotated[str, typer.Option("--pretrained")] = ...,  # "google/flan-t5-large",
        data_dir: Annotated[str, typer.Option("--data_dir")] = "data/GNER",
        data_config_dir: Annotated[str, typer.Option("--data_config_dir")] = None,  # "configs/dataset/ZSE",
        instruct_file: Annotated[str, typer.Option("--instruct_file")] = None,  # "configs/instruction/GNER-paper.json",
        train_file: Annotated[str, typer.Option("--train_file")] = None,  # "data/GNER/pile-ner.jsonl",
        eval_file: Annotated[str, typer.Option("--eval_file")] = None,  # "data/GNER/ZSE-validation.jsonl,
        pred_file: Annotated[str, typer.Option("--pred_file")] = None,  # "data/GNER/ZSE-test.jsonl",
        use_cache_data: Annotated[bool, typer.Option("--use_cache_data/--no_use_cache_data")] = False,
        progress_seconds: Annotated[float, typer.Option("--progress_seconds")] = 10.0,
        max_source_length: Annotated[int, typer.Option("--max_source_length")] = 640,
        max_target_length: Annotated[int, typer.Option("--max_target_length")] = 640,
        max_generation_tokens: Annotated[int, typer.Option("--max_generation_tokens")] = 640,
        ignore_pad_token_for_loss: Annotated[bool, typer.Option("--ignore_pad_token_for_loss/--no_ignore_pad_token_for_loss")] = True,
        write_predictions: Annotated[bool, typer.Option("--write_predictions/--no_write_predictions")] = True,
        # for Seq2SeqTrainingArguments
        generation_num_beams: Annotated[int, typer.Option("--generation_num_beams")] = 1,
        use_flash_attention: Annotated[bool, typer.Option("--use_flash_attention/--no_use_flash_attention")] = False,
        gradient_checkpointing: Annotated[bool, typer.Option("--gradient_checkpointing/--no_gradient_checkpointing")] = True,
        per_device_train_batch_size: Annotated[int, typer.Option("--per_device_train_batch_size")] = 1,
        gradient_accumulation_steps: Annotated[int, typer.Option("--gradient_accumulation_steps")] = 1,
        per_device_eval_batch_size: Annotated[int, typer.Option("--per_device_eval_batch_size")] = 25,
        eval_accumulation_steps: Annotated[int, typer.Option("--eval_accumulation_steps")] = 1,
        num_train_epochs: Annotated[float, typer.Option("--num_train_epochs")] = 1,
        logging_epochs: Annotated[float, typer.Option("--logging_epochs")] = -1,
        eval_at_step: Annotated[int, typer.Option("--eval_at_step")] = 10,
        eval_epochs: Annotated[float, typer.Option("--eval_epochs")] = -1,
        save_epochs: Annotated[float, typer.Option("--save_epochs")] = -1,
        save_total_limit: Annotated[int, typer.Option("--save_total_limit")] = 3,
        metric_for_best_model: Annotated[str, typer.Option("--metric_for_best_model")] = None,
        logging_steps: Annotated[int, typer.Option("--logging_steps")] = 1,
        eval_steps: Annotated[int, typer.Option("--eval_steps")] = -1,
        save_steps: Annotated[int, typer.Option("--save_steps")] = -1,
        eval_delay: Annotated[float, typer.Option("--eval_delay")] = 0.0,
        max_steps: Annotated[int, typer.Option("--max_steps")] = -1,
        report_to: Annotated[str, typer.Option("--report_to")] = "tensorboard",  # "tensorboard" or "none",  # tensorboard --bind_all --logdir output/GNER
        lr_scheduler_type: Annotated[str, typer.Option("--lr_scheduler_type")] = "constant",
        warmup_ratio: Annotated[float, typer.Option("--warmup_ratio")] = 0.0,
        warmup_steps: Annotated[int, typer.Option("--warmup_steps")] = 0,
        learning_rate: Annotated[float, typer.Option("--learning_rate")] = 5e-5,
        # for DeepSpeed
        trainer_deepspeed: Annotated[str, typer.Option("--trainer_deepspeed")] = None,  # for deepspeed.launcher.runner
        accelerate_deepspeed: Annotated[bool, typer.Option("--accelerate_deepspeed")] = False,  # for accelerate.commands.launch
        # for FSDP
        fsdp: Annotated[bool, typer.Option("--fsdp/--no_fsdp")] = False,
        fsdp_transformer_layer_cls_to_wrap: Annotated[str, typer.Option("--fsdp_transformer_layer_cls_to_wrap")] = None,  # "T5Block"
        # for ProjectEnv
        run_version: Annotated[str, typer.Option("--run_version")] = None,
        output_name: Annotated[str, typer.Option("--output_name")] = "GNER",
        output_home: Annotated[str, typer.Option("--output_home")] = "output-lfs",
        output_file: Annotated[str, typer.Option("--output_file")] = "train-metrics.csv",
        logging_file: Annotated[str, typer.Option("--logging_file")] = "train-loggings.out",
        argument_file: Annotated[str, typer.Option("--argument_file")] = "train-arguments.json",
        random_seed: Annotated[int, typer.Option("--random_seed")] = 7,
        max_workers: Annotated[int, typer.Option("--max_workers")] = int(os.cpu_count() / 4),
        world_size: Annotated[int, typer.Option("--world_size")] = -1,
        local_rank: Annotated[int, typer.Option("--local_rank")] = -1,
        debugging: Annotated[bool, typer.Option("--debugging/--no_debugging")] = False,
        verbose: Annotated[int, typer.Option("--verbose")] = 1,
):
    # Setup project environment
    if local_rank < 0 and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 0 and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    stamp = now_stamp()
    stamp = sorted(gather_object([stamp]))[0]
    env = NewProjectEnv(
        time_stamp=from_timestamp(stamp, fmt='%m%d-%H%M%S'),
        local_rank=local_rank,
        world_size=world_size,
        run_version=run_version,
        output_name=output_name,
        output_home=output_home,
        output_file=new_path(output_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_file=new_path(logging_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        argument_file=new_path(argument_file, post=from_timestamp(stamp, fmt='%m%d-%H%M%S')),
        logging_level=logging.WARNING,
        logging_format=LoggingFormat.CHECK_24,
        random_seed=random_seed,
        max_workers=1 if debugging else max(max_workers, 1),
        debugging=debugging,
    )

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=DeepSpeedPlugin() if accelerate_deepspeed else None,
    )
    accelerator.wait_for_everyone()

    # Setup training arguments
    level_to_str = {n: s for s, n in transformers.utils.logging.log_levels.items()}
    log_level_str = level_to_str.get(env.logging_level, "passive")
    log_level_rep_str = level_to_str.get(env.logging_level + 10, "passive")
    args = TrainingArgumentsForAccelerator(
        env=env,
        data=CustomDataArguments(
            data_dir=data_dir,
            data_config_dir=data_config_dir,
            instruct_file=instruct_file,
            train_file=train_file,
            eval_file=eval_file,
            pred_file=pred_file,
            pretrained=pretrained,
            use_cache_data=use_cache_data,
            progress_seconds=progress_seconds,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            max_generation_tokens=max_generation_tokens,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            write_predictions=write_predictions,
        ),
        train=ExSeq2SeqTrainingArguments(
            generation_num_beams=generation_num_beams,
            predict_with_generate=True,
            remove_unused_columns=False,
            overwrite_output_dir=True,
            output_dir=str(env.output_dir),
            report_to=report_to,
            log_level=log_level_str,
            log_level_replica=log_level_rep_str,
            use_flash_attention=use_flash_attention,
            gradient_checkpointing=gradient_checkpointing,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            num_train_epochs=num_train_epochs,
            logging_epochs=logging_epochs,
            eval_at_step=eval_at_step,
            eval_epochs=eval_epochs,
            save_epochs=save_epochs,
            save_total_limit=save_total_limit,
            metric_for_best_model=metric_for_best_model,
            load_best_model_at_end=bool(metric_for_best_model),
            logging_strategy="steps" if logging_steps >= 1 else "epoch" if logging_epochs == 1 else "no",
            eval_strategy="steps" if eval_steps >= 1 else "epoch" if eval_epochs == 1 else "no",
            save_strategy="steps" if save_steps >= 1 else "epoch" if save_epochs == 1 else "no",
            logging_steps=logging_steps if logging_steps >= 1 else sys.maxsize,
            eval_steps=eval_steps if eval_steps >= 1 else sys.maxsize,
            save_steps=save_steps if save_steps >= 1 else sys.maxsize,
            eval_delay=eval_delay,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.0,
            tf32=is_torch_tf32_available(),
            bf16=is_torch_bf16_gpu_available(),
            bf16_full_eval=is_torch_bf16_gpu_available(),
            local_rank=env.local_rank,
            deepspeed=trainer_deepspeed,
            fsdp="full_shard auto_wrap" if fsdp else "",
            fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
            seed=env.random_seed,
            disable_tqdm=True,
        ),
    )
    args.env.local_rank = args.train.local_rank
    accelerator.wait_for_everyone()

    # Setup logging
    process_log_level = args.train.get_process_log_level()
    args.env.setup_logger(process_log_level)
    datasets_set_verbosity(process_log_level)
    transformers_set_verbosity(process_log_level)
    set_verbosity_warning("torch")
    set_verbosity_info("c10d-NullHandler-default")
    if accelerator.is_main_process:
        if debugging:
            set_verbosity_debug("chrisdata", "gner")
        else:
            set_verbosity_info("chrisbase", "gner")

    # Set random seed
    set_seed(args.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("medium")
    accelerator.wait_for_everyone()

    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}",
                  rt=1, rb=1, rc='=', verbose=verbose, args=args):
        # Load pretrained config
        config = AutoConfig.from_pretrained(
            args.data.pretrained,
            use_cache=not args.train.gradient_checkpointing,
            trust_remote_code=True,
        )
        is_encoder_decoder = config.is_encoder_decoder

        # Load pretrained tokenizer
        if is_encoder_decoder:
            tokenizer = AutoTokenizer.from_pretrained(
                args.data.pretrained,
                trust_remote_code=True,
            )
        else:  # https://github.com/Vision-CAIR/MiniGPT-4/issues/129
            tokenizer = AutoTokenizer.from_pretrained(
                args.data.pretrained,
                padding_side="left",
                add_eos_token=True,
                add_bos_token=True,
                trust_remote_code=True,
            )
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token  # https://medium.com/@rschaeffer23/how-to-fine-tune-llama-3-1-8b-instruct-bf0a84af7795
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
            # tokenizer.add_special_tokens({'pad_token': "<pad>"})  # https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token

        # Load pretrained model
        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.data.pretrained,
                from_tf=bool(".ckpt" in str(args.data.pretrained)),
                config=config,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.data.pretrained,
                from_tf=bool(".ckpt" in str(args.data.pretrained)),
                config=config,
                device_map="cuda" if args.train.use_flash_attention else None,
                torch_dtype="auto" if args.train.use_flash_attention else None,
                attn_implementation="flash_attention_2" if args.train.use_flash_attention else config._attn_implementation,
                trust_remote_code=True,
            )
        model.generation_config.pad_token_id = tokenizer.pad_token_id  # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        accelerator.wait_for_everyone()
        logger.info(f"model type: {type(model)}")
        logger.info(f"model pad_token_id: {model.generation_config.pad_token_id}")
        if args.train.use_flash_attention:
            logger.info(f"model attn_implementation: {config._attn_implementation} -> {model.config._attn_implementation}")

        # Load dataset by data configuration or data file
        raw_datasets = {}
        if args.data.data_config_dir:
            raw_datasets = load_dataset(
                "gner/gner_dataset.py",
                data_dir=args.data.data_dir,
                data_config_dir=args.data.data_config_dir,
                instruction_file=args.data.instruct_file,
                add_dataset_name=False,
                trust_remote_code=True,
            )
            # raw_datasets.cleanup_cache_files()
            for split in args.data.data_files:
                if split in raw_datasets and len(raw_datasets[split]) > 0:
                    if accelerator.is_main_process:
                        backup_file = f"{args.data.data_dir}/{args.data.data_config_dir.name}-{split}.jsonl"
                        raw_datasets[split].to_json(backup_file, lines=True, force_ascii=False)
                        logger.info(f'Loaded raw {split} dataset by {args.data.data_config_dir / f"{split}_configs.json"}: {len(raw_datasets[split])} samples -> {backup_file}')
        for split in args.data.data_files:
            if split not in raw_datasets or len(raw_datasets[split]) == 0:
                if args.data.data_file(split):
                    raw_datasets[split] = load_dataset("json", data_files=str(args.data.data_file(split)), split="train")
                    logger.info(f"Loaded raw {split} dataset by {args.data.data_file(split)}: {len(raw_datasets[split])} samples")
        accelerator.wait_for_everyone()

        # Preprocess dataset as model inputs
        tokenized_datasets = {}
        for split in args.data.data_files:
            tokenized_datasets[split] = preprocess_dataset(
                split=split,
                raw_datasets=raw_datasets,
                is_encoder_decoder=is_encoder_decoder,
                max_source_length=args.data.max_source_length,
                max_target_length=args.data.max_target_length,
                tokenizer=tokenizer,
                local_rank=args.train.local_rank,
                max_workers=args.env.max_workers,
                progress_seconds=args.data.progress_seconds / 5,
                cache_file=args.data.cache_file(
                    split=split,
                    data_size=len(raw_datasets[split]),
                    tokenizer_path=tokenizer.name_or_path,
                ) if args.data.use_cache_data else None,
            )
        args.train.do_train, args.train.do_eval, args.train.do_predict = (
            tokenized_datasets[x] and len(tokenized_datasets[x]) > 0
            for x in (datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST)
        )
        logger.info(f"args.train.do_train   = {args.train.do_train}")
        logger.info(f"args.train.do_eval    = {args.train.do_eval}")
        logger.info(f"args.train.do_predict = {args.train.do_predict}")
        accelerator.wait_for_everyone()

        # Data collator
        label_pad_token_id = -100 if args.data.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = gner.DataCollatorForGNER(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8 if args.train.fp16 else None,
            label_pad_token_id=label_pad_token_id,
            return_tensors="pt",
        )

        # Set compute_metrics function
        compute_metrics = lambda *vs, **ws: eval_predictions(
            *vs, **ws, write_predictions=args.data.write_predictions, accelerator=accelerator,
        ) if args.train.predict_with_generate else None

        # Initialize trainer
        trainer = gner.GNERTrainer(
            args=args.train,
            model=model,
            train_dataset=tokenized_datasets[datasets.Split.TRAIN],
            eval_dataset=tokenized_datasets[datasets.Split.VALIDATION],
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            is_encoder_decoder=is_encoder_decoder,
            max_generation_tokens=max_generation_tokens,
        )
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(gner.CustomProgressCallback(
            trainer=trainer,
            metric_file=args.env.output_dir / args.env.output_file,
            logging_epochs=args.train.logging_epochs,
            eval_at_step=args.train.eval_at_step,
            eval_epochs=args.train.eval_epochs if args.train.do_eval else 0,
            save_epochs=args.train.save_epochs,
            progress_seconds=args.data.progress_seconds,
            metric_formats={
                "epoch": ".2f",
                "loss": ".6f",
                "train_loss": ".4f",
                "eval_average": ".4f",
                "total_pflos": ".3f",
            },
        ))
        accelerator.wait_for_everyone()

        # do_train
        if args.train.do_train:
            train_result: TrainOutput = trainer.train()
            with patch("builtins.print", side_effect=lambda *xs: logger.info(*xs)):
                trainer.log_metrics("train", train_result.metrics)
                trainer.save_metrics("train", train_result.metrics)
            convert_all_events_in_dir(args.train.output_dir)

        # do_eval
        if args.train.do_eval:
            eval_result: Dict[str, float] = trainer.evaluate(tokenized_datasets[datasets.Split.VALIDATION], metric_key_prefix="eval")
            with patch("builtins.print", side_effect=lambda *xs: logger.info(*xs)):
                trainer.log_metrics("eval", eval_result)
                trainer.save_metrics("eval", eval_result)

        # do_predict
        if args.train.do_predict:
            pred_result: PredictionOutput = trainer.predict(tokenized_datasets[datasets.Split.TEST], metric_key_prefix="pred")
            with patch("builtins.print", side_effect=lambda *xs: logger.info(*xs)):
                trainer.log_metrics("pred", pred_result.metrics)
                trainer.save_metrics("pred", pred_result.metrics)

    accelerator.end_training()


if __name__ == "__main__":
    AppTyper.run(main)
