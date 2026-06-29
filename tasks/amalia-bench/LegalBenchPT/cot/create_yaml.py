categories_dict = {
    "Direito_Administrativo": "Administrative",
    "Direito_Aéreo": "Aviation",
    "Direito_Bancário": "Banking",
    "Direito_Comercial": "Commercial",
    "Direito_Constitucional": "Constitutional",
    "Direito_Financeiro": "Financial",
    "Direito_Fiscal": "Tax",
    "Direito_Internacional_Privado": "Private International",
    "Direito_Internacional_Público": "Public International",
    "Direito_Marítimo": "Maritime",
    "Direito_Penal": "Criminal",
    "Direito_Processual_Administrativo": "Administrative Procedure",
    "Direito_Processual_Civil": "Civil Procedure",
    "Direito_Processual_Penal": "Criminal Procedure",
    "Direito_Processual_do_Trabalho": "Labor Procedure",
    "Direito_da_Concorrência": "Competition",
    "Direito_da_Energia": "Energy",
    "Direito_da_Família": "Family",
    "Direito_da_Insolvência": "Insolvency",
    "Direito_da_UE_e_Comunitário": "EU and Community",
    "Direito_das_Finanças_Públicas": "Public Finance",
    "Direito_das_Obrigações": "Obligations",
    "Direito_das_Sociedades_Comerciais": "Corporate",
    "Direito_das_Sucessões": "Succession",
    "Direito_do_Ambiente": "Environmental",
    "Direito_do_Trabalho": "Labor",
    "Direito_do_Urbanismo": "Urban Planning",
    "Direito_dos_Contratos": "Contract",
    "Direito_dos_Transportes": "Transportation",
    "Direito_dos_Valores_Mobiliários": "Securities",
    "Direitos_Reais": "Property",
}


groups_dict = {
    "Public": [
            "Direito_Administrativo",
            "Direito_Constitucional",
            "Direito_Financeiro",
            "Direito_Fiscal",
            "Direito_Penal",
            "Direito_Processual_Administrativo",
            "Direito_Processual_Civil",
            "Direito_Processual_Penal",
            "Direito_Processual_do_Trabalho",
            "Direito_da_Energia",
            "Direito_das_Finanças_Públicas",
            "Direito_do_Ambiente",
            "Direito_do_Urbanismo",
        ],
    "Private": [
        "Direito_Aéreo",
        "Direito_Bancário",
        "Direito_Comercial",
        "Direito_Internacional_Privado",
        "Direito_Marítimo",
        "Direito_da_Família",
        "Direito_da_Insolvência",
        "Direito_das_Obrigações",
        "Direito_das_Sociedades_Comerciais",
        "Direito_das_Sucessões",
        "Direito_do_Trabalho",
        "Direito_dos_Contratos",
        "Direito_dos_Transportes",
        "Direito_dos_Valores_Mobiliários",
        "Direitos_Reais",
        ],
    "Public_Private": [
        "Direito_da_Concorrência",
        ],
    "Public International G": [
        "Direito_Internacional_Público",
        ],
    "EU and Community G": [
        "Direito_da_UE_e_Comunitário",
        ]
  }


types = {
    "Multiple Choice",
    "Cloze Tasks",
    "True/False",
    "Case Analysis Questions",
    "Multiple Selection Questions",
    "Matching Questions",
}


yaml_category_type_pair = """dataset_path: BeatrizCanaverde/LegalBench.PT
dataset_name: {category_pt}
test_split: test
task: amalia_legalbench_pt_{category}_{type}_cot
task_alias: {category}_{type}_cot
#tag: amalia_legalbench_pt_{category}
doc_to_text: Question
doc_to_target: Answer
process_docs: !function utils.process_{type}
output_type: generate_until
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 2000
metric_list:
  - metric: !function utils.legal_metric
    aggregation: !function utils.legal_metric_agg
    higher_is_better: true
metadata:
  version: 0.0"""


yaml_category = """group: amalia_legalbench_pt_{category}_cot
group_alias: {category}_cot
task: {list_pairs_category_type}
aggregate_metric_list:
  - metric: legal_metric
    weight_by_size: true
metadata:
  version: 0.0"""
    #aggregation: !function utils.custom_aggregate
    #higher_is_better: true


yaml_group = """group: amalia_legalbench_pt_{group}_cot
group_alias: {group}_cot
task: {list_categories}
aggregate_metric_list:
  - metric: legal_metric
    weight_by_size: true
metadata:
  version: 0.0"""
    #aggregation: !function utils.custom_aggregate
    #higher_is_better: true


yaml_total = """group: amalia_legalbench_pt_cot
task: 
  - {list_groups}
aggregate_metric_list:
  - metric: legal_metric
    weight_by_size: true
metadata:
  version: 0.0"""
    #aggregation: !function utils.custom_aggregate
    #higher_is_better: true




import os, re
from datasets import load_dataset


task_folder = os.path.dirname(os.path.abspath(__file__))


for category in categories_dict.keys():
    
    dataset = load_dataset("BeatrizCanaverde/LegalBench.PT", category)
    subset_data = dataset["test"]

    # Convert Hugging Face dataset to pandas DataFrame
    subset_data  = subset_data .to_pandas()

    list_pairs_category_type = ""

    for type in types:

      subset = subset_data[subset_data["Type"] == type]
      if not subset.empty:
          list_pairs_category_type += f"\n  - amalia_legalbench_pt_{categories_dict[category].replace(" ", "_").lower()}_{re.sub(r"[ /]", "_", type).lower()}_cot"

      category_type_content = yaml_category_type_pair.format(category_pt=category, category=categories_dict[category].replace(" ", "_").lower(), type=re.sub(r"[ /]", "_", type).lower())
      category_type_filename = os.path.join(task_folder, f'CT_legalbench_pt_{categories_dict[category].replace(" ", "_").lower()}_{re.sub(r"[ /]", "_", type).lower()}.yaml')
      
      with open(category_type_filename, 'w') as yaml_file:
          yaml_file.write(category_type_content)

    category_content = yaml_category.format(category=categories_dict[category].replace(" ", "_").lower(), list_pairs_category_type=list_pairs_category_type)
    category_filename = os.path.join(task_folder, f'C_legalbench_pt_{categories_dict[category].replace(" ", "_").lower()}.yaml')
    
    with open(category_filename, 'w') as yaml_file:
        yaml_file.write(category_content)




list_groups = ""

for group in groups_dict.keys():
    
    if not list_groups:
        list_groups += f"\n  - amalia_legalbench_pt_{group.replace(" ", "_").lower()}_cot"

    list_categories = ""

    for category in groups_dict[group]:
        list_categories += f"\n  - amalia_legalbench_pt_{categories_dict[category].replace(" ", "_").lower()}_cot"

    group_content = yaml_group.format(group=group.replace(" ", "_").lower(), list_categories=list_categories)
    group_filename = os.path.join(task_folder, f'G_legalbench_pt_{group.replace(" ", "_").lower()}.yaml')
    
    with open(group_filename, 'w') as yaml_file:
        yaml_file.write(group_content)




list_groups = "\n  - ".join([f"amalia_legalbench_pt_{group.replace(" ", "_").lower()}_cot" for group in groups_dict.keys()])

total_content = yaml_total.format(list_groups=list_groups)
total_filename = os.path.join(task_folder, f'legalbench_pt.yaml')

with open(total_filename, 'w') as yaml_file:
  yaml_file.write(total_content)









