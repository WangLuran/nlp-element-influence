import numpy as np
import copy
import scipy.special as special

"""Required data structure: 
 data is a dictionary with keys: context_probs, paraphrases_probs, answers
 data[prob_label] is a tensor with shape: num_datapoints * num_questions * num_paraphrases * num_options
 data[answer] is a tensor with shape: num_datapoints * num_questions
"""

# Calculate model accuracy
def get_accuracy(datas):
    r = 0
    n = 0
    for data in datas:
        for i in range(len(data['paraphrases_probs'])):
            if i < 60:
                continue
            for j in range(8):
                for k in range(len(data['paraphrases_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    answer = data['answers'][i][k]
                    if np.argmax(np.array(p_o)) == answer:
                        r += 1
                    n += 1
    return r/n


#Calibration
def anneal_probs_np(datas):
    logits = []
    probs = []

    #get current model probs and avg max prob
    for data in datas:
        for i in range(len(data['paraphrases_probs'])):
            for j in range(8):
                for k in range(len(data['context_probs'][i])):
                    p_o = data['paraphrases_probs'][i][k][j][0]
                    logits.append(np.log(np.array(p_o)))
                    probs.append(p_o)
    logits = np.array(logits)
    probs = np.array(probs)
    max_probs = [max(i) for i in probs]
    avg_prob = np.mean(max_probs)

    #look at current model accuracy
    acc = get_accuracy(datas)
    print(acc)

    #do the annealing
    alpha = 1
    while avg_prob > acc:  
        alpha += 0.001
        annealed_logits = logits/alpha
        probs_np  = special.softmax(annealed_logits, axis=-1)
        max_probs = [max(i) for i in probs_np]
        avg_prob = np.mean(max_probs)

    return alpha