import argparse
import os
import sys
import json
import time
import openai


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--data_path', type=str, help='Load path of data')
parser.add_argument('--save_path', type=str, help='Load path to save results with increased distractors')
parser.add_argument('--prompts', default=None,type=list, help='Prompts used to generate paraphrases')


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
 
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as f:
            expanded_examples = json.load(f)

        # Discard incomplete data
        if expanded_examples[-1]['paraphrases'] == []:
            expanded_examples = expanded_examples[0:-1]
    
    else:
        expanded_examples = []


    # Load starting point to avoid repetition
    start_point = len(expanded_examples)

    # The 8 prompts used to generate paraphrases with different readability levels
    if args.prompts == None:
        prompt_lists = ['Paraphrase this document for a professional. It should be extremely difficult to read and best understood by university graduates.','Paraphrase this document for college graduate level (US). It should be very difficult to read and best understood by university graduates.','Paraphrase this document for college level (US). It should be difficult to read.','Paraphrase this document for 10th-12th grade school level (US). It should be fairly difficult to read.','Paraphrase this document for 8th/9th grade school level (US). It should be plain English and easily understood by 13- to 15-year-old students.','Paraphrase this document for 7th grade school level (US). It should be fairly easy to read.','Paraphrase this document for 6th grade school level (US). It should be easy to read and conversational English for consumers.','Paraphrase this document for 5th grade school level (US). It should be very easy to read and easily understood by an average 11-year old student.']
    else:
        prompt_lists = args.prompts

    # Load dataset
    with open(args.data_path,'r') as f:
        data = json.load(f)

    # Load openai api
    openai.api_key = args.api_key
    
    count = 0
    ids_pass = []

    for ex in data:
        id_c = ex['context_id']
        context = ex['context']
        question = ex['question']
        answer = ex['label']
        options = ex['options']

        # skip the contexts has been paraphrased
        if id_c not in ids_pass:
            ids_pass.append(id_c)

            # update generated data and generate new datapoints
            if len(ids_pass) > start_point: 
                with open(args.save_path, 'w') as f:
                    json.dump(expanded_examples, f)
                print("Saved up to:", len(expanded_examples))
                print("----------------------")

                curr_example = {'context': context, 'paraphrases':[], 'questions': [], 'options':[],'answer': []}
                expanded_examples.append(curr_example)
                expanded_examples[-1]['questions'].append(question)
                expanded_examples[-1]['options'].append(options)
                expanded_examples[-1]['answer'].append(answer)

                # Generate paraphrses
                paraphrases = []
                for prompt_start in prompt_lists:
                    prompt = prompt_start + 'Document: '+context
                    model = "gpt-3.5-turbo"
                    response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
                    paraphrase = response.choices[0].message.content.strip()
                    paraphrases.append(paraphrase)
                expanded_examples[-1]['paraphrases'] = paraphrases
            continue

        if id_c in ids_pass:
            if len(ids_pass) > start_point:
                expanded_examples[-1]['questions'].append(question)
                expanded_examples[-1]['options'].append(options)
                expanded_examples[-1]['answer'].append(answer)
            continue

        if len(ids_pass) <= start_point:
            continue

        '''# Generate paraphrses
        paraphrases = []
        for prompt_start in prompt_lists:
            prompt = prompt_start + 'Document: '+context
            model = "gpt-3.5-turbo"
            response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
            paraphrase = response.choices[0].message.content.strip()
            paraphrases.append(paraphrase)
        expanded_examples[-1]['paraphrases'] = paraphrases'''

    # Update the last datapoint
    with open(args.save_path, 'w') as f:
        json.dump(expanded_examples, f)
    print("Saved up to:", len(expanded_examples))
    print("----------------------")


if __name__ == '__main__':
    args = parser.parse_args()

    for count in range(1,20):
        try:
            main(args)
            time.sleep(1)
            break
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.Timeout:
            print("openai.error.TimeoutError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("openai.error.ServiceUnavailableError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
