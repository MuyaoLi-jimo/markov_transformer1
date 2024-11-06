import logging
import os
from contextlib import nullcontext
from PIL import Image
import pathlib
import json
import numpy as np
from torchvision import transforms
import random

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"
    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

################
# Create a data collator to encode text and image pairs
################

class ChatDataCollator:
    def __init__(self, tokenizer, max_seq_length = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __call__(self, examples):
        texts = []

        for example in examples:
            # if len(example["images"]) > 1:
            #     raise ValueError("This collator only supports one image per example")
            messages = example["conversations"]
            processed_messages = []
            for message in messages:
                processed_message_role = message["role"]
                processed_message_content = ""
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_message_content += item["text"]
                    else:
                        print("Only text is supported for now.")
                processed_messages.append({"role": processed_message_role, "content": processed_message_content})
            
            text = self.tokenizer.apply_chat_template(
                processed_messages, tokenize=False, add_generation_prompt=False
            )
            # print("text: ", text)
            texts.append(text)

        batch = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=self.max_seq_length, truncation=True)

        labels = batch["input_ids"].clone()
        # TODO: add back 
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


if __name__ == "__main__":
    model_name_or_path = 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    data_collator = ChatDataCollator(tokenizer, max_seq_length = max_seq_length)  #预处理数据的工具
    
    exit()
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure use_cache is set to False
    model.config.use_cache = False

    data_collator = ChatDataCollator(tokenizer, max_seq_length = training_args.max_seq_length)  #预处理数据的工具

    ################
    # Dataset
    ################
    # raw_datasets = load_dataset(sft_script_args.dataset_name)
    
    # raw_datasets = load_dataset("json", data_files=sft_script_args.dataset_name, split = "train")
    # train_dataset = raw_datasets['train']
    # eval_dataset = raw_datasets['test']
    train_dataset_file = sft_script_args.dataset_name + "-train.json"
    eval_dataset_file = sft_script_args.dataset_name + "-valid.json"
    if not os.path.exists(train_dataset_file) or not os.path.exists(eval_dataset_file):
        print("train and eval datasets not found!")
        # raise ValueError("The dataset files do not exist.")
        if os.path.exists(sft_script_args.dataset_name+".json"):
            print("Loading the dataset from json file...")
            with open(sft_script_args.dataset_name+".json", "r") as f:
                data = json.load(f)
        elif os.path.exists(sft_script_args.dataset_name+".jsonl"):
            print("Loading the dataset from jsonl file...")
            data = []
            with open(sft_script_args.dataset_name+".jsonl", "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError("The dataset files do not exist.")

        print("Shuffling and splitting the dataset into train and eval datasets...")
        random.shuffle(data)
        test_samples = min(len(data)*0.02, 1000)
        test_data = data[:test_samples]
        train_data = data[test_samples:]
        with open(train_dataset_file, "w") as f:
            json.dump(train_data, f)
        with open(eval_dataset_file, "w") as f:
            json.dump(test_data, f)

    raw_datasets = load_dataset("json", data_files={"train": train_dataset_file, "validation": eval_dataset_file})
    
    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']
    
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        training_args.resume_from_checkpoint = True
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",  # need a dummy field, UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`.
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True}
        )

    # trainer.train(resume_from_checkpoint = training_args.resume_from_checkpoint)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)