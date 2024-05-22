import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import wandb
import time

load_dotenv()

model_name = "apple/OpenELM-270M"
tokenizer_name = "meta-llama/Llama-2-7b-hf" if model_name == "apple/OpenELM-270M" else model_name
max_context_length = 8096
dataset_name = 'HuggingFaceTB/cosmopedia'
dataset_subset = "openstax"
dataset_test_split_ratio = 0.05
output_dir_tmpl = "/bucket/openelm-cosmopedia-openstax-{timestamp}"
lora_r = 16
lora_alpha = 32
lora_targets = ["qkv_proj", "out_proj", "proj_1", "proj_2"]
run_name_tmpl = "openelm-cosmopedia-openstax-{timestamp}"


# Hyperparameters
num_epochs = 2
lr = 2e-4
logging_steps = 50
eval_steps = 1000
save_steps = 250


# Modify for prompt templates
def generate_prompt(system: str, user: str, assistant: str):
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]\n\n{assistant}</s>".strip()
    return prompt


def setup_logging():
    wandb.login(key=os.getenv("WANDB_API_KEY"))


def train():
    # Metadata
    timestamp = str(int(time.time()))
    run_name = run_name_tmpl.format(timestamp=timestamp)
    output_dir = output_dir_tmpl.format(timestamp=timestamp)

    # Loading the Model and Convert to a PEFT Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16,
                                                 quantization_config=bnb_config, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    # Function for Tokenizing the Dataset
    def generate_and_tokenize_prompt(example):
        full_prompt = generate_prompt(
            "You are a helful digital assistant. Please provide safe, ethical and accurate information to the user.",
            example["prompt"],
            example["text"],
        )
        tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True, max_length=max_context_length)
        return tokenized_full_prompt

    # Loading the Dataset
    dataset = load_dataset(dataset_name, dataset_subset, split="train")
    split_dataset = dataset.train_test_split(test_size=dataset_test_split_ratio)
    train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]
    train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt, num_proc=16)
    test_dataset = test_dataset.map(generate_and_tokenize_prompt, num_proc=16)

    # Setup Logging
    setup_logging()

    # Setup Training
    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        bf16=True,
        save_total_limit=4,
        eval_strategy="steps",
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        output_dir=output_dir,
        save_strategy='steps',
        save_steps=save_steps,
        report_to="wandb",
        run_name=run_name
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    # Training
    trainer.train()

    # Post-Processing
    wandb.finish()
