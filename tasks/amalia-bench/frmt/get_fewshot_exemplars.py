from datasets import load_dataset
import os


def process_for_fewshot(dataset, lang):

    filtered = dataset.filter(lambda x: x[lang] is not None and x[lang] != "N/A" and x[lang] != "")

    if lang == "pt":
        filtered = filtered.remove_columns(["br"])
    elif lang == "br":
        filtered = filtered.remove_columns(["pt"])

    filtered = filtered.rename_column(lang, "target_lang")
    lexical_filtered = filtered.filter(lambda x: x["bucket"] == "lexical")
    
    return lexical_filtered


# Main script to load, process, and save the dataset
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = "hugosousa/frmt"
    dataset = load_dataset(dataset_path)

    exemplars_split = dataset["exemplars"]
    processed_pt_dataset = process_for_fewshot(exemplars_split, "pt")
    pt_save_path = os.path.join(script_dir, "fewshot_exemplars_pt")
    processed_pt_dataset.save_to_disk(pt_save_path)

    processed_br_dataset = process_for_fewshot(exemplars_split, "br")
    br_save_path = os.path.join(script_dir, "fewshot_exemplars_br")
    processed_br_dataset.save_to_disk(br_save_path)


if __name__ == "__main__":
    main()
