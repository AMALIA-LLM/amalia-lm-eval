from datasets import load_dataset
import os


labels_dict = {
    "PT-PT": 0,
    "PT-BR": 1,
    "PT": 2
}


def process_for_fewshot(dataset, lang):
    return dataset.filter(lambda x: x["label"] == labels_dict[lang])


# Main script to load, process, and save the dataset
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_path = "LCA-PORVID/dsl_tl"
    dataset = load_dataset(dataset_path)
    dataset = dataset["train"]
    
    processed_ptpt_dataset = process_for_fewshot(dataset, "PT-PT")
    ptpt_save_path = os.path.join(script_dir, "train_ptpt")
    processed_ptpt_dataset.save_to_disk(ptpt_save_path)

    processed_ptbr_dataset = process_for_fewshot(dataset, "PT-BR")
    ptbr_save_path = os.path.join(script_dir, "train_ptbr")
    processed_ptbr_dataset.save_to_disk(ptbr_save_path)

    processed_pt_dataset = process_for_fewshot(dataset, "PT")
    pt_save_path = os.path.join(script_dir, "train_pt")
    processed_pt_dataset.save_to_disk(pt_save_path)


if __name__ == "__main__":
    main()
