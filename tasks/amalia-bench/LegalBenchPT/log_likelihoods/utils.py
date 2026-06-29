from functools import partial
import re


def process_question(example):
    if example["Type"] == "True/False":
        end = "O output deverá ser apenas 'A afirmação é verdadeira' ou 'A afirmação é falsa'"
        return example["Question"][: -len(end)] + "Resposta: A afirmação é "
    else:
        end = "O output deverá ser apenas: 'A resposta correta é: {letra}'"
        return example["Question"][: -len(end)] + "Resposta: A opção correta é "
    
    
def process_docs(dataset, type):
    return dataset.filter(lambda x: x["Type"] == type)

process_multiple_choice = partial(process_docs, type="Multiple Choice")
process_cloze_tasks = partial(process_docs, type="Cloze Tasks")
process_true_false = partial(process_docs, type="True/False")
process_case_analysis_questions = partial(process_docs, type="Case Analysis Questions")



def process_answer(example):
    if example["Type"] == "True/False":
        answer = example["Answer"]
        if answer == "Verdadeiro":
            return 0 #"verdadeira"
        else:
            return 1 #"falsa"
    else:
        #import pdb; pdb.set_trace()
        answer = example["Answer"]
        if len(answer) > 1:
            print(f"Example: {example}\n\n")
        return ord(answer) - ord('a')




def separate_question_and_options(text):

    # Find the start of the relevant section
    start_idx = -1
    for marker in ["\n\n\nPergunta:\n", "\n\n\nAfirmacao:\n"]:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break

    if start_idx == -1:
        # Marker not found, fallback to whole text
        lines = text.split("\n")
    else:
        lines = text[start_idx:].split("\n")

    # Pattern to detect the options
    option_pattern = r"^\s*[a-z][\)\.]\s*"

    # Find the index where options start
    options_start_idx = None
    for i, line in enumerate(lines):
        if re.match(option_pattern, line.strip(), re.IGNORECASE):
            options_start_idx = i
            break
    if options_start_idx is None:
        return ["a", "b", "c", "d"]

    # The remaining lines are the options
    options = lines[options_start_idx:]
    answer_options = []
    for option in options:
        match = re.match(r"^\s*([a-zA-Z]+)[\)\.]\s*", option.strip())
        if match:
            label = match.group(1).strip().lower()
            if label:
                answer_options.append(label)

    # Remove duplicates and empty strings, preserve order
    answer_options = list(dict.fromkeys([a for a in answer_options if a]))

    if not answer_options:
        return ["a", "b", "c", "d"]
    else:
        return answer_options


def get_answer_options(example):
    if example["Type"] == "True/False":
        return ["verdadeira", "falsa"]
    else:
        return separate_question_and_options(example["Question"])