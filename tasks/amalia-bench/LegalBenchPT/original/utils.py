import re

def process_results(doc, results):
    """
    Processes the model's output and compares it with the reference answer.
    """
    generated = results[0].strip()
    question_type = doc.get('Type')

    if question_type in ["Multiple Choice", "Cloze Tasks", "Case Analysis Questions"]:
        pattern = r'A resposta correta é:?\s*([A-Za-z])'
        match = re.search(pattern, generated, re.IGNORECASE)
        if match:
            pred = match.group(1).upper()
            return {'accuracy': int(pred == doc['Answer'].upper())}
        else:
            return {'accuracy': 0}

    elif question_type == "True/False":
        pattern = r'A afirmação é \s*(falsa|verdadeira)'
        match = re.search(pattern, generated, re.IGNORECASE)
        if match:
            pred = match.group(1).lower()
            return {'accuracy': int(pred == doc['Answer'].lower())}
        else:
            return {'accuracy': 0}

    elif question_type == "Matching Questions":
        pattern = re.compile(r'([a-zA-Z])\s*[-)]?\s*(\d)|(\d)\s*[-)]?\s*([a-zA-Z])')
        matches = pattern.findall(generated)
        preds = set()
        for m in matches:
            if m[0] and m[1]:
                preds.add((m[0].upper(), m[1]))
            elif m[2] and m[3]:
                preds.add((m[3].upper(), m[2]))
        gold = set(tuple(pair) for pair in doc['Answer'])
        correct = preds & gold
        return {'accuracy': len(correct) / len(gold) if gold else 0}

    elif question_type == "Multiple Selection Questions":
        pattern = re.compile(r'[^A-Za-z]([A-Za-z])\)')
        matches = pattern.findall(generated)
        preds = set(m.upper() for m in matches)
        gold = set(ans.upper() for ans in doc['Answer'])
        correct = preds & gold
        return {'accuracy': len(correct) / len(gold) if gold else 0}

    else:
        return {'accuracy': 0}



def accuracy(doc, results):
    """
    Computes accuracy for a single document.
    """
    return process_results(doc, results)['accuracy']



def mean(metrics):
    """
    Aggregates a list of metric values by computing their mean.
    """
    return sum(metrics) / len(metrics) if metrics else 0.0




from functools import partial


def process_docs(dataset, type):
    #if type == "Matching Questions":
    #    return dataset.filter(lambda x: x["Type"] == type).map(lambda x: {**x, "Question": x["Question"] + "A resposta correta é: "})
    if type == "Multiple Selection Questions":
        return dataset.filter(lambda x: x["Type"] == type).map(lambda x: {**x, "Question": x["Question"]})  # + "Resposta: "})
    elif type == "True/False":
        return dataset.filter(lambda x: x["Type"] == type).map(lambda x: {**x, "Question": x["Question"]})  # + "Resposta: A afirmação é: "})
    else:
        #return dataset.filter(lambda x: x["Type"] == type)
        return dataset.filter(lambda x: x["Type"] == type).map(lambda x: {**x, "Question": x["Question"]})  # + "A resposta correta é: "})



process_multiple_choice = partial(process_docs, type="Multiple Choice")
process_cloze_tasks = partial(process_docs, type="Cloze Tasks")
process_true_false = partial(process_docs, type="True/False")
process_case_analysis_questions = partial(process_docs, type="Case Analysis Questions")
process_multiple_selection_questions = partial(process_docs, type="Multiple Selection Questions")
process_matching_questions = partial(process_docs, type="Matching Questions")






def prepend_name(responses):
    return [["Multiple choice:\n\n" + resp for resp in instance_resps] for instance_resps in responses]




from typing import List, Dict

def legal_avg(input):
    #a passthrough function for aggregate metrics
    #print(f"input:\n{input}\n\n")
    return input

def delete_element(lst, number):
    for i,s in enumerate(lst):
        if number == s:
            if i==0:
                return lst[1:]
            elif i==len(lst)-1:
                return lst[:-1]
            else:
                return lst[:i]+lst[i+1:]    
            
def evaluate_f1(GUESS, TRUTH):
    tp = 0
    fp = 0
    fn = 0
    for a in GUESS:
        if a in TRUTH:
            TRUTH = delete_element(TRUTH, a)
            tp += 1
        else:
            fp += 1
    fn = len(TRUTH)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    if precision + recall == 0:
        return 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
def unique_list(original_list):
    unique = []
    for i in original_list:
        if i not in unique:
            unique.append(i)
    return unique


def legal_metric(items):
    return items


def legal_metric_agg(input):
    #print(f"input:\n{input}\n\n")
    correct_answers = [sublist[0].strip() for sublist in input]
    #print(f"correct_answers:\n{correct_answers}\n\n")
    generations = [sublist[1].strip() for sublist in input]
    #print(f"generations:\n{generations}\n\n")
    type_answer = correct_answers[0]

    if len(type_answer) == 1:
        return legal_avg_acc_multiple_choice(correct_answers, generations)

    if type_answer in ["Verdadeiro", "Falso"]:
        return legal_avg_acc_true_false(correct_answers, generations)

    if type_answer.startswith("[") and type_answer.endswith("]"):
        num_open = type_answer.count("[")
        num_close = type_answer.count("]")

        if num_open >= 2 and num_close >= 2:
            return legal_avg_f1_agg_matching_questions(correct_answers, generations)
        else:
            return legal_avg_f1_agg_multiple_selection_questions(correct_answers, generations)


import ast

def legal_avg_f1_agg_matching_questions(correct_answers, generations):
    #correct_answers = ["".join(sublist[0]) for sublist in input]
    #generations = [sublist[1][0] for sublist in input]
    tot_f1 = 0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        pattern = re.compile(r'([a-zA-Z])\s*[-)]?\s*(\d)|(\d)\s*[-)]?\s*([a-zA-Z])')
        matches = pattern.findall(predicted_answer)
        guess = []
        #print(f"type: {type(target_answer)}\n")
        target_answer = ast.literal_eval(target_answer)   
        for match in matches:
            letter1, digit, digit2, letter2 = match
            if letter1 and digit:
                guess.append([letter1.lower(), digit])
            elif digit2 and letter2:
                guess.append([letter2.lower(), digit2])
        guess = unique_list(guess)  
        #n = len(target_answer)
        #target_answer = ast.literal_eval(target_answer)     
        target_answer = sorted(target_answer)
        guess = sorted(guess)
        #guess = sorted(guess)[:n+1]
        #print(f"predicted_answer: {predicted_answer}\n")
        #print(f"target_answer: {target_answer}\n")
        #print(f"guess: {guess}\n\n")
        tot_f1 += evaluate_f1(guess, target_answer)
    avg_f1 = tot_f1/len(correct_answers) if len(correct_answers) > 0 else 0
    return avg_f1


def legal_avg_f1_agg_multiple_selection_questions(correct_answers, generations):
    #correct_answers = ["".join(sublist[0]) for sublist in input]
    #generations = [sublist[1][0] for sublist in input]
    tot_f1 = 0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        #pattern_cut = re.compile(r'[^A-Za-z]([A-Za-z]\))\s*[A-Za-z]{2,}')
        #match_cut = pattern_cut.search(predicted_answer)
        #print(f"predicted_answer: {predicted_answer}\n")
        #if match_cut:
        #    cut_position = match_cut.end()
        #    predicted_answer = predicted_answer[:cut_position]
        pattern = re.compile(r'[^A-Za-z]([A-Za-z])\)')
        #print(f"cut_answer: {predicted_answer}\n")
        #print(f"target_answer: {target_answer}\n")
        guess = pattern.findall(predicted_answer) #.lower())
        guess = list(set(guess))
        target_answer = ast.literal_eval(target_answer)
        target_answer = sorted([i.lower() for i in target_answer])
        guess = sorted([i.lower() for i in guess])
        #print(f"guess: {guess}\n\n")
        tot_f1 += evaluate_f1(guess, target_answer)
    avg_f1 = tot_f1/len(correct_answers) if len(correct_answers) > 0 else 0
    return avg_f1


# from sklearn.metrics import balanced_accuracy_score

def legal_avg_acc_multiple_choice(correct_answers, generations):
    #correct_answers = ["".join(sublist[0]) for sublist in input]
    #generations = [sublist[1][0] for sublist in input]
    tot_acc = 0
    #guess_answers = []
    #correct_answers = [i.lower() for i in correct_answers]
    for predicted_answer, target_answer in zip(generations, correct_answers):
        pattern = r'A resposta correta é:?\s*([A-Za-z])'  # r'([A-Za-z])'     # r'A resposta correta é:?\s*([A-Za-z])'  # r'A resposta correta é:?\s*([A-Za-z])'
        match = re.search(pattern, predicted_answer)
        #print(f"predicted_answer: {predicted_answer}\n")
        #print(f"target_answer: {target_answer}\n")
        if match:
            guess = match.group(1)
        else:
            guess = ""
            #print(f"No match for predicted_answer: {predicted_answer}")  # <--- DEBUG here
        #print(f"guess: {guess}\n\n")
        #guess_answers.append(guess.lower())
        if target_answer.lower() == guess.lower():
            tot_acc += 1
    #bal_acc = balanced_accuracy_score(correct_answers, guess_answers)
    #num_questions = len(correct_answers)
    avg_acc = tot_acc/len(correct_answers) if len(correct_answers) > 0 else 0
    return avg_acc  # avg_acc  # bal_acc


def legal_avg_acc_true_false(correct_answers, generations):
    #correct_answers = ["".join(sublist[0]) for sublist in input]
    #generations = [sublist[1][0] for sublist in input]
    tot_acc = 0.0
    for predicted_answer, target_answer in zip(generations, correct_answers):
        pattern = r'A afirmação é \s*(falsa|verdadeira)'  # r'(fals|verdadeir)'    # r'A afirmação é \s*(fals|verdadeir)'  # r'A afirmação é \s*(falsa|verdadeira)'
        match = re.search(pattern, predicted_answer)
        #print(f"predicted_answer: {predicted_answer}\n")
        #print(f"target_answer: {target_answer}\n")
        if match:
            guess = match.group(1).lower()
        else:
            guess = ""
        if guess == "falsa":
            guess = "Falso"
        elif guess == "verdadeira":
            guess = "Verdadeiro"
        # if guess == "fals":
        #     guess = "Falso"
        # elif guess == "verdadeir":
        #     guess = "Verdadeiro"
        #print(f"guess: {guess}\n\n")
        if target_answer == guess:
            tot_acc += 1
    avg_acc = tot_acc/len(correct_answers) if len(correct_answers) > 0 else 0
    return avg_acc


def legal_mean(arr):
    return sum(arr) / len(arr)

    

def group_avg(input):
    #a passthrough function for aggregate metrics
    return input

def group_avg_agg(input):
    #a passthrough function for aggregate metrics
    return input

def custom_aggregate(results):
    """
    Computes a weighted average of subset scores.

    Args:
        results (dict): A dictionary containing evaluation results for each subset.

    Returns:
        float: The weighted average score.
    """
    total_score = 0.0
    total_weight = 0

    for subset_name, subset_data in results.items():
        # Extract the score and weight for each subset
        score = subset_data.get('score', 0.0)
        weight = subset_data.get('weight', 1)  # Default weight is 1 if not specified

        total_score += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0  # Avoid division by zero

    return total_score / total_weight

