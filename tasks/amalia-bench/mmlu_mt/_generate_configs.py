# noqa
"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

SUBJECTS_PT_PT = {
    "abstract_algebra": "álgebra abstrata",
    "anatomy": "anatomia",
    "astronomy": "astronomia",
    "business_ethics": "ética empresarial",
    "clinical_knowledge": "conhecimento clínico",
    "college_biology": "biologia universitária",
    "college_chemistry": "química universitária",
    "college_computer_science": "ciência da computação universitária",
    "college_mathematics": "matemática universitária",
    "college_medicine": "medicina universitária",
    "college_physics": "física universitária",
    "computer_security": "segurança informática",
    "conceptual_physics": "física conceptual",
    "econometrics": "econometria",
    "electrical_engineering": "engenharia eletrotécnica",
    "elementary_mathematics": "matemática elementar",
    "formal_logic": "lógica formal",
    "global_facts": "factos globais",
    "high_school_biology": "biologia do ensino secundário",
    "high_school_chemistry": "química do ensino secundário",
    "high_school_computer_science": "ciência da computação do ensino secundário",
    "high_school_european_history": "história europeia do ensino secundário",
    "high_school_geography": "geografia do ensino secundário",
    "high_school_government_and_politics": "governo e política do ensino secundário",
    "high_school_macroeconomics": "macroeconomia do ensino secundário",
    "high_school_mathematics": "matemática do ensino secundário",
    "high_school_microeconomics": "microeconomia do ensino secundário",
    "high_school_physics": "física do ensino secundário",
    "high_school_psychology": "psicologia do ensino secundário",
    "high_school_statistics": "estatística do ensino secundário",
    "high_school_us_history": "história dos EUA do ensino secundário",
    "high_school_world_history": "história mundial do ensino secundário",
    "human_aging": "envelhecimento humano",
    "human_sexuality": "sexualidade humana",
    "international_law": "direito internacional",
    "jurisprudence": "jurisprudência",
    "logical_fallacies": "falácias lógicas",
    "machine_learning": "aprendizagem automática",
    "management": "gestão",
    "marketing": "marketing",
    "medical_genetics": "genética médica",
    "miscellaneous": "miscelânea",
    "moral_disputes": "disputas morais",
    "moral_scenarios": "cenários morais",
    "nutrition": "nutrição",
    "philosophy": "filosofia",
    "prehistory": "pré-história",
    "professional_accounting": "contabilidade profissional",
    "professional_law": "direito profissional",
    "professional_medicine": "medicina profissional",
    "professional_psychology": "psicologia profissional",
    "public_relations": "relações públicas",
    "security_studies": "estudos de segurança",
    "sociology": "sociologia",
    "us_foreign_policy": "política externa dos EUA",
    "virology": "virologia",
    "world_religions": "religiões mundiais",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            # description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"
            # pt-pt version
            pt_subject = SUBJECTS_PT_PT[subject]
            description = f"As seguintes são perguntas de escolha múltipla (com respostas) sobre {pt_subject}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"mmlu_{args.task_prefix}_{category}"
            if args.task_prefix != ""
            else f"mmlu_{category}",
            "task": f"mmlu_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"mmlu_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_subcategories = [
            f"mmlu_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_subcategories = [f"mmlu_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_{args.task_prefix}"
                if args.task_prefix != ""
                else "mmlu",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
