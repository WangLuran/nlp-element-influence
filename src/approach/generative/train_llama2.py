#! /usr/bin/env python

"""
This script finetunes a Llama-2 model on the RACE++ multiple-choice reading comprehension dataset.
"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from typing import Union, Callable, Tuple
from collections import namedtuple
from datasets import Dataset
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup

from src.utils.load_datasets import load_mcrc_data

from peft import get_peft_model, TaskType, PeftConfig, PeftModel
from peft import PromptTuningConfig, PromptTuningInit
from peft import LoraConfig, PrefixTuningConfig

#== General Util tools ============================================================================#
CAUSAL_LLMS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "gpt2": "gpt2"
}

SEQ2SEQ_LLMS = {}

MODEL_NAME_TO_PATH = {**CAUSAL_LLMS, **SEQ2SEQ_LLMS}
MODEL_PATH_TO_NAME = {v:k for k, v in MODEL_NAME_TO_PATH.items()}

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--seed', type=int, help='Seed for training regime')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--max_length', type=int, default=1024, help='Maximum number of tokens')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Set learning rate')
parser.add_argument('--num_epochs', type=int, default=3, help='Total number of epochs for training')
parser.add_argument('--lora_r', type=int, default=32, help='Decomposition matrix size for lora')
parser.add_argument('--data_path', type=str, help='Load path to the RACE++ train data')
parser.add_argument('--save_dir', type=str, help='Load directory path to save trained model')

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

#== Model utility functions ======================================================================#
def is_seq2seq(model_path):
    if model_path in SEQ2SEQ_LLMS.values():
        output = True
    elif model_path in CAUSAL_LLMS.values():
        output = False
    else:
        raise ValueError('invalid model name')
    return output


#== Data Preprocessing ============================================================================#

def process_mcrc_dataset(dataset_name:str, model_name:str, data_path):
    prompt_template = '{context}\n\n{question}\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}'

    if not is_seq2seq(model_name):
        prompt_template = prompt_template + '\n\nAnswer:'

    # load data
    # train, val, test = load_mcrc_data(dataset_name)
    train = load_mcrc_data('race_plus_plus', data_path)

    # process data
    LABEL_WORDS = ['A', 'B', 'C', 'D']
    train_data = []
    # val_data = []
    # test_data = []

    # for split, output in [(train, train_data), (val, val_data), (test, test_data)]:
    for split, output in [(train, train_data)]:
        for ex in split:
            input_text = prompt_template.format(
                context=ex.context, 
                question=ex.question,
                option_1=ex.options[0],
                option_2=ex.options[1],
                option_3=ex.options[2],
                option_4=ex.options[3]
            )
            label_text = LABEL_WORDS[ex.label]
            output.append(({'text':input_text, 'label_text':label_text, 'label_id':ex.label}))

    train_dataset = Dataset.from_list(train_data)
    # val_dataset = Dataset.from_list(val_data)
    # test_dataset = Dataset.from_list(test_data)
    # return train_dataset,  val_dataset, test_dataset
    return train_dataset

#== Tokenizer functions ==========================================================================#
def create_train_preprocess_function(tokenizer, max_length)->Callable[[dict], dict]:
    def preprocess_function(examples):
        inputs = [x for x in examples['prompt']]
        targets = [x for x in examples['label']]
        batch_size = len(inputs)

        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos since concatenation
        
        #tokenize input and label text, and prepare for training
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100]*len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1]*len(model_inputs["input_ids"][i])
        
        #truncating and padding to max length
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            
            # do padding
            pad_length = max_length-len(sample_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id]*pad_length
            model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [0]*pad_length
            labels["input_ids"][i] = label_input_ids + [-100]*pad_length
            
            # truncate
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        
        model_inputs["labels"] = labels["input_ids"]       
        return model_inputs
    return preprocess_function

def tokenize_dataset(dataset:Dataset, preprocess_function:Callable, num_proc:int=10)->Tuple[Dataset]:
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        remove_columns=dataset.column_names,
    )
    return processed_dataset

def filter_dataset(tok_dataset:Dataset, max_length:int=None, size:int=None, num_proc:int=10)->Dataset:
    #remove inputs that are too long
    if max_length:
        filter_fn = lambda example: not all([x == -100 for x in example['labels']])
        tok_dataset = tok_dataset.filter(filter_fn, num_proc=num_proc)
    
    #reduce traing set if shorter sizes provided
    if size and size < len(tok_dataset):
        rng1 = random.Random(42)
        random_numbers = rng1.sample(range(0, len(tok_dataset)), size)
        tok_dataset = tok_dataset.select(list(random_numbers))
    
    return tok_dataset

#== Transformer Model Functions ===================================================================#
def create_tokenizer(model_path:str)->AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def create_peft_config(peft_strategy:str, model_path:str, N:int=None)->PeftConfig:    
    task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq(model_path) else TaskType.CAUSAL_LM
    peft_config = LoraConfig(
        task_type=task_type, inference_mode=False, r=N, lora_alpha=2*N, lora_dropout=0.1
    )
    return peft_config

def create_base_model(model_path:str, dtype:str)->Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    D_TYPES = {'bfloat16':torch.bfloat16, 'float32':torch.float32}
    torch_dtype = D_TYPES[dtype]

    if model_path in CAUSAL_LLMS.values():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    elif model_path in SEQ2SEQ_LLMS.values():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        raise ValueError('invalid model')
    return model

def create_model(model_path:str, peft_config:str, dtype:str=None)->Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    model = create_base_model(model_path, dtype)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# Main
def main(args):

    # Preliminaries
    seed_val = args.seed
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = get_default_device()


    model_name = "llama2-7b-chat"
    model_path = MODEL_NAME_TO_PATH[model_name]
    tokenizer = create_tokenizer(model_path)
    peft_config = create_peft_config('lora', model_path, args.lora_r)
    model = create_model(model_path, peft_config, 'bfloat16')

    print("Model initialisation complete.")


    # prepare dataset
    train_dataset = process_mcrc_dataset('race_plus_plus', model_path, args.data_path)
    
    # tokenize training datasets
    train_prc_fn = create_train_preprocess_function(tokenizer, args.max_length)
    tok_train_dataset = tokenize_dataset(train_dataset, train_prc_fn)
    tok_train_dataset = filter_dataset(tok_train_dataset, args.max_length)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)