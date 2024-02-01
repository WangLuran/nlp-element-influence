"""
This script includes functions to calculate influence of components of the input on model outputs.
"""
"""Required data structure: 
 data is a dictionary with keys: context_probs, paraphrases_probs, answers
 data[prob_label] is a tensor with shape: num_datapoints * num_questions * num_paraphrases * num_options
 data[answer] is a tensor with shape: num_datapoints * num_questions
"""

import numpy as np
import copy
import scipy.special as special

#== Basic functions ============================================================================#

# KL divergence
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# entropy calculation
def get_entropy(list):
    out = 0
    list = list.tolist()
    ds = []
    for i in range(len(list)):
        if list[i] < 0.000000001:
            ds.append(list[i])
    if len(ds) >= 3:
        return 0
    for d in ds:
        list.remove(d)
    for i in list:
        out -= np.log(i)*i
    return out



#== Influence calculation ==========================================================================#


# whole input influence
"""input: datas: a list of dictionary
   alpha: the parameter for calibration"""
def uncertainty_score_totals(datas,alpha):
    uncertainty = 0
    et = 0
    count = 0
    for data in datas:
        for i in range(len(data['context_probs'])):
            for j in range(8):
                for k in range(len(data['context_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j])

                    #reorder the options to make the first option always the right option
                    tcp = copy.deepcopy(data['paraphrases_probs'][i][k][j][data['answers'][i][k]])
                    exchenged = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    p_o[0] = tcp
                    p_o[data['answers'][i][k]] = exchenged
                    p_o = np.array(p_o)

                    #calibration
                    p_o  = special.softmax(np.log(p_o)/alpha, axis=-1)

                    entropy_e = get_entropy(p_o)
                    uncertainty += entropy_e
                    et += p_o
                    count += 1
    et /= count
    uncertainty /= count
    uncertainty_m = get_entropy(et) - uncertainty

    return uncertainty_m


# context influence
def uncertainty_score_contexts(datas,alpha):
    uncertainty = 0
    et = 0
    count = 0
    u_count = 0
    for data in datas:
        for i in range(len(data['context_probs'])):
            prob_sum = 0
            for j in range(data['paraphrases_probs'][i][k]):
                prob_sum = 0
                for k in range(len(data['context_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j])

                    #reorder the options to make the first option always the right option
                    tcp = copy.deepcopy(data['paraphrases_probs'][i][k][j][data['answers'][i][k]])
                    exchenged = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    p_o[0] = tcp
                    p_o[data['answers'][i][k]] = exchenged
                    p_o = np.array(p_o)

                    #calibration 
                    p_o  = special.softmax(np.log(p_o)/alpha, axis=-1)

                    prob_sum += p_o/len(data['context_probs'][i])
                    et += p_o
                    count += 1

                entropy_e = get_entropy(prob_sum)
                u_count += 1
                uncertainty += entropy_e                
    et /= count
    uncertainty /= u_count
    uncertainty_m = get_entropy(et) - uncertainty 

    return uncertainty_m


#question influence
def uncertainty_score_questions(datas,alpha):
    uncertainty = 0
    count = 0
    u_count = 0
    et = 0
    for data in datas:
        for i in range(len(data['context_probs'])):
            prob_sum = 0
            for j in range(data['paraphrases_probs'][i][k]):
                prob_sum = 0
                for k in range(len(data['context_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j])

                    #reorder the options to make the first option always the right option
                    tcp = copy.deepcopy(data['paraphrases_probs'][i][k][j][data['answers'][i][k]])
                    exchenged = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    p_o[0] = tcp
                    p_o[data['answers'][i][k]] = exchenged
                    p_o = np.array(p_o)

                    #calibration
                    p_o  = special.softmax(np.log(p_o)/alpha, axis=-1)

                    entropy_e = get_entropy(p_o)
                    uncertainty += entropy_e
                    u_count += 1
                    prob_sum += p_o/len(data['context_probs'][i])
                et += get_entropy(prob_sum)
                count += 1
    et /= count
    uncertainty /= u_count
    uncertainty_m = et - uncertainty

    return uncertainty_m

# paraphrasing influence
def uncertainty_score_paras(datas,alpha):
    uncertainty = 0
    et = 0
    count = 0
    u_count = 0
    for data in datas:
        for i in range(len(data['context_probs'])):
            prob_sum = 0
            for j in range(data['paraphrases_probs'][i][k]):
                prob_sum_u = 0
                for k in range(len(data['context_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j])

                    #reorder the options to make the first option always the right option
                    tcp = copy.deepcopy(data['paraphrases_probs'][i][k][j][data['answers'][i][k]])
                    exchenged = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    p_o[0] = tcp
                    p_o[data['answers'][i][k]] = exchenged
                    p_o = np.array(p_o)

                    #calibration
                    p_o  = special.softmax(np.log(p_o)/alpha, axis=-1)

                    prob_sum += p_o/(len(data['context_probs'][i])*data['paraphrases_probs'][i][k])
                    prob_sum_u += p_o/len(data['context_probs'][i])
                entropy_e = get_entropy(prob_sum_u)
                uncertainty += entropy_e
                u_count += 1
            et += get_entropy(prob_sum)
            count += 1
    et /= count
    uncertainty /= u_count
    uncertainty_m = et - uncertainty 

    return uncertainty_m

# semantic meaning influence
def uncertainty_score_meanings(datas,alpha):
    uncertainty = 0
    et = 0
    count = 0
    u_count = 0
    for data in datas:
        for i in range(len(data['context_probs'])):
            prob_sum = 0
            for j in range(data['paraphrases_probs'][i][k]):
                for k in range(len(data['context_probs'][i])):
                    p_o = copy.deepcopy(data['paraphrases_probs'][i][k][j])

                    #reorder the options to make the first option always the right option
                    tcp = copy.deepcopy(data['paraphrases_probs'][i][k][j][data['answers'][i][k]])
                    exchenged = copy.deepcopy(data['paraphrases_probs'][i][k][j][0])
                    p_o[0] = tcp
                    p_o[data['answers'][i][k]] = exchenged
                    p_o = np.array(p_o)

                    #calibration
                    p_o  = special.softmax(np.log(p_o)/alpha, axis=-1)

                    prob_sum += p_o/(len(data['context_probs'][i])*8)
                    et += p_o
                    count += 1
            entropy_e = get_entropy(prob_sum)
            uncertainty += entropy_e
            u_count += 1
    et /= count
    uncertainty /= u_count
    uncertainty_m = get_entropy(et) - uncertainty 

    return uncertainty_m

