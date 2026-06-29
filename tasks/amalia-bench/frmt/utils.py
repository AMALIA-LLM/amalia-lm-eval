from functools import partial
from datasets import load_dataset
import os


def format_nshot_example(example, including_exemplars=True):

    #prompt += example["en"] + "\n"
    prompt = ""

    if including_exemplars:
        prompt += "Inglês: " + example["en"].strip() + "\nPortuguês: " + example["target_lang"].strip()
        return prompt

    else:
        prompt += "Inglês: " + example["en"].strip() + "\nPortuguês: "
        return prompt


#doc_to_text_pt = partial(format_nshot_example, including_exemplars=False, lang="pt")
doc_to_text = partial(format_nshot_example, including_exemplars=False)
#fewshot_to_text_pt = partial(format_nshot_example, including_exemplars=True, lang="pt")
fewshot_to_text = partial(format_nshot_example, including_exemplars=True)



def doc_to_target(example):
    return example["target_lang"]


doc_to_target_pt = partial(doc_to_target)
doc_to_target_br = partial(doc_to_target)




def process_docs(dataset, subject, lang):
    #import pdb
    #pdb.set_trace()
    filtered = dataset.filter(lambda x: x[lang] is not None and x[lang] != "N/A" and x[lang] != "")
    if lang == "pt":
        filtered = filtered.remove_columns(["br"])
    elif lang == "br":
        filtered = filtered.remove_columns(["pt"])
    filtered = filtered.rename_column(lang, "target_lang")
    return filtered.filter(lambda x: x["bucket"] == subject)   #.select(range(5))  # DELETE SELECT!!


process_lexical_pt = partial(process_docs, subject="lexical", lang="pt")
process_entity_pt = partial(process_docs, subject="entity", lang="pt")
process_random_pt = partial(process_docs, subject="random", lang="pt")

process_lexical_br = partial(process_docs, subject="lexical", lang="br")
process_entity_br = partial(process_docs, subject="entity", lang="br")
process_random_br = partial(process_docs, subject="random", lang="br")




def process_for_fewshot(lang):
    dataset = load_dataset("amalia-llm/frmt_fewshot", split=f"fewshot_exemplars_{lang}")
    #dataset = dataset.sort("en")
    return dataset


process_for_fewshot_pt = partial(process_for_fewshot, lang="pt")
process_for_fewshot_br = partial(process_for_fewshot, lang="br")





def process_results(doc, results):
    """
    Strips leading/trailing spaces from all model outputs.
    Args:
        - doc: the document containing the input and output (keys are the variables from the prompt_template)
        - results: the output of the model (still dunnow why its a list but it doesn't depend on the batchsize)
    Returns:
        - a dictionary containing the metrics (metric_name: metric_value)
    """
    #import pdb; pdb.set_trace()
    # Strip spaces from all model outputs
    #cleaned_results = [r.strip() for r in results]
    # You can return the cleaned results, or use them as needed
    #return {"chrf": cleaned_results}

    completion = results[0]
    completion = completion.strip().split('\n', 1)[0]

    model_input, reference_out = doc["en"], doc["target_lang"]
    items = [reference_out, completion, model_input]
    #items = [
    #    i if i is not None else "" for i in items
    #]

    return {
    #    "bleu": items[:2],
        "chrf": items[:2],
    #    "comet_22_da": items,
    #    "bleurt": items[:2],
    }