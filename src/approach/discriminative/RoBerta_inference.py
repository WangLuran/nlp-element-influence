# Default pretrained Roberta is from https://huggingface.co/LIAMF-USP/roberta-large-finetuned-race

import argparse
import os
import sys
import json
import torch
import numpy as np

from transformers import RobertaTokenizer
from transformers import  RobertaForMultipleChoice


MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_path', type=str, help='Load path of inference data')
parser.add_argument('--save_path',default='', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--model_path', default=None,type=str, help='Load path to which trained model is saved')
parser.add_argument('--tokenizer_path', default=None,type=str, help='Load path to which trained model is saved')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    # Load data
    with open(args.data_path) as f:
        inference_data = json.load(f)


    # Initialize model for inference
    model_path = args.model_path
    if model_path == None:
        tokenizer = RobertaTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        model = RobertaForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
        model.eval().to(device)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
        model = RobertaForMultipleChoice.from_pretrained(args.model_path)
        model.eval().to(device)
    
    print("Model initialisation complete.")


    # input encoding
    def prepare_answering_input(tokenizer, question, options,context,max_seq_length=512,):
        c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
        c_plus_q_4 = [context] * len(options)
        question_options = []
        for option in options:
            if question.find("_") != -1:
            # fill in the banks questions
                question_option = question.replace("_", option)
            else:
                question_option = question + " " + option
            question_options.append(question_option)
        tokenized_examples = tokenizer(c_plus_q_4,question_options,add_special_tokens=True,padding="max_length",truncation=True,return_overflowing_tokens=False, max_length=max_seq_length,return_tensors="pt")
        input_ids = tokenized_examples['input_ids'].unsqueeze(0).to(device)
        attention_mask = tokenized_examples['attention_mask'].unsqueeze(0).to(device)  
        example_encoded = {"input_ids": input_ids,"attention_mask": attention_mask}
        return example_encoded

    """
    Input data structure:
    A list of dictionaries with keys: context, paraphases, questions, answers, options
    """
    """
    Output data structure: 
    data is a dictionary with keys: context_probs, paraphrases_probs, answers
    data[prob_label] is a tensor with shape: num_datapoints * num_questions * num_paraphrases * num_options
    data[answer] is a tensor with shape: num_datapoints * num_questions
    """

    context_score_all = [] 
    paraphrases_scores_all = [] 
    answers_all = [] 
    num_right = 0
    num_total = 0

    for i, ex in enumerate(inference_data):
        # To avoid failure in cuda out of memory
        try:
            context = ex['context']
            questions = ex['questions']
            answers = ex['answer']
            options_s = ex['options']
            paras = ex['paraphrases']
            context_score_all.append([])
            paraphrases_scores_all.append([])
            answers_all.append([])
            for j in range(len(questions)):
                scores_paras = []
                question = questions[j]
                options = options_s[j]
                answer = answers[j] 
                inputs = prepare_answering_input(tokenizer=tokenizer, question=question,options=options, context=context)
                outputs = model(**inputs)
                predc = outputs[0].detach().cpu().numpy()
                probs_c = np.exp(predc)/np.sum(np.exp(predc))
                if np.argmax(probs_c) == answer:
                    num_right += 1
                num_total += 1
                probs_c = probs_c.tolist()
                context_score_all[-1].append(probs_c[0])
                for para in paras:
                    inputs = prepare_answering_input(tokenizer=tokenizer, question=question,options=options, context=para)
                    outputs = model(**inputs)
                    predc = outputs[0].detach().cpu().numpy()
                    probs = np.exp(predc)/np.sum(np.exp(predc))
                    probs = probs.tolist()
                    scores_paras.append(probs)
                paraphrases_scores_all[-1].append(scores_paras[0])
                answers_all[-1].append(answer)
            print(str(i+1)+'/'+str(len(inference_data))+ ' done')
        except torch.cuda.OutOfMemoryError:
            context_score_all = context_score_all[0:-1]
            paraphrases_scores_all = paraphrases_scores_all[0:-1]
            answers_all = answers_all[0:-1]
            continue
    expanded_examples = {'context_probs':context_score_all,'paraphrases_probs':paraphrases_scores_all,'answers':answers_all}
    
    print('accuracy is: ',num_right/num_total)
    
    with open(args.save_path, 'w') as f:
        json.dump(expanded_examples, f)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
