# Loading various MCRC datasets

import json
from datasets import load_dataset

def asNum(x):
    if x=="A":
        return 0
    if x=="B":
        return 1
    if x=="C":
        return 2
    if x=="D":
        return 3

def get_race_plus_plus(split, path):
    # The split is not important here as the path will be for the appropriate directory
    with open(f"{path}middle.json") as f:
        middle_data = json.load(f)
    with open(f"{path}high.json") as f:
        high_data = json.load(f)
    with open(f"{path}college.json") as f:
        college_data = json.load(f)
    all_data = [middle_data, high_data, college_data]
    difficulty = ['easy', 'medium', 'hard']
    flat_data = []

    for sub_count, sub_data in enumerate(all_data):
        for context_id, item in enumerate(sub_data):
            context = item["article"]
            questions = item["questions"]
            options = item["options"]
            answers = item["answers"]
            for qu_num, question in enumerate(questions):
                opts = options[qu_num]
                lab = asNum(answers[qu_num])
                curr = {'context':context, 'question':question, 'options':opts, 'label':lab, 'context_id':context_id, 'difficulty':difficulty[sub_count]}
                flat_data.append(curr)
    return flat_data


def get_mctest(split, path):
    # Here the dataset is gotten from huggingface
    data = load_dataset('sagnikrayc/mctest',split=split)
    flat_data = []

    options_labels = ['A','B','C','D']
    for item in data:
        context = item["story"]
        question = item["question"]
        opts = [item["answer_options"][i] for i in options_labels]
        lab = asNum(item["answer"])
        context_id = item['idx']['story']
        curr = {'context':context, 'question':question, 'options':opts, 'label':lab, 'context_id':context_id}
        flat_data.append(curr)
    return flat_data


def get_camchoice(split, path):
    # The split is not important here as the path will be for the appropriate directory
    with open(f"{path}.json") as f:
        data = json.load(f)
    flat_data = []

    for item in data:
        context = item["context"]
        questions = item["questions"]
        context_id = item['cotext_num']
        difficulty_level = item['target_level']
        for question_dict in questions:
            opts = question_dict['options']
            lab = question_dict['answer']
            question = question_dict['question']
            curr = {'context':context, 'question':question, 'options':opts, 'label':lab, 'context_id':context_id, 'difficulty':difficulty_level}
            flat_data.append(curr)
    return flat_data


def load_mcrc_data(dataset_name, split, path=None):

    if dataset_name == 'race_plus_plus':
        all_data = get_race_plus_plus(split, path)
    elif dataset_name == 'mctest':
        all_data = get_mctest(split, path) 
    elif dataset_name == 'camchoice':
        all_data = get_camchoice(split, path)
    return all_data
