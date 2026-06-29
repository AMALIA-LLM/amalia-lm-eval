dataset_to_pretty_name_dict = {
        'arc_challenge_chat_mt_pt': 'ARC-C',
        #'arc_challenge_mt_pt': 'ARC-C',
        "arc_challenge_mt_pt": "ARC-C (base)",
        'mmlu_mt_pt_chat': 'MMLU',
        'mmlu_mt_pt': 'MMLU (base)',
        'piqa-mt-pt': 'PiQA',
        'global_piqa_completions_por_latn_port': 'G-PiQA-c',
        'global_piqa_prompted_por_latn_port': 'G-PiQA-p',
        'gsm8k_cot_mt_pt': 'GSM8K',
        'gsm8k_cot_llama_mt_pt': 'GSM8K (llama)',
        'aime_pt': 'AIME',
        'aime24_pt': 'AIME-24',
        'aime25_pt': 'AIME-25',
        'minerva_math500_pt': 'MATH 500 (Minerva)',
        'minerva_math_pt': 'MATH Hendrycks (Minerva)',
        'math_pt_br_mcq': 'MATH-PT MCQ (br)',
        'math_pt_pt_mcq': 'MATH-PT MCQ (pt)',
        'math_pt_pt_mcq_no_figures': 'MATH-PT MCQ (pt-no-fig)',
        'pt_exams_chat': 'PT-E',
        'pt_exams': 'PT-E (base)',
        'pt_exams_portuguese_chat': 'PT-E Portuguese',
        'pt_exams_mathematics_a_chat': 'PT-E Math',
        'pt_exams_history_a_chat': 'PT-E History',
        'pt_exams_philosophy_chat': 'PT-E Philosophy',
        'pt_exams_geography_chat': 'PT-E Geography',
        'pt_exams_bio_geo_chat': 'PT-E Bio/Geo',
        'pt_exams_cot': 'PT-E (CoT)',
        'pt_exams_cot_portuguese': 'PT-E Portuguese (CoT)',
        'pt_exams_cot_mathematics_a': 'PT-E Math (CoT)',
        'pt_exams_cot_history_a': 'PT-E History (CoT)',
        'pt_exams_cot_philosophy': 'PT-E Philosophy (CoT)',
        'pt_exams_cot_geography': 'PT-E Geography (CoT)',
        'pt_exams_cot_bio_geo': 'PT-E Bio/Geo (CoT)',
        'pt_text_completion_chat': 'PT-C',
        'pt_text_completion': 'PT-C (base)',
        'hellaswag_mt_pt': 'HelSw',
        'commonsense_qa-mt-pt': 'CSQA',
        'openbookqa-mt-pt': 'OBQA',
        'winogrande-mt-pt': 'WG',
        'social_iqa-mt-pt': 'SiQA',
        'truthfulqa_mc2_mt_pt': 'TQA',
        'truthfulqa_mc1_mt_pt': 'TQA (mc1)',
        'frmt_fewshot': 'FRMT',
        'frmt_fewshot_pt': 'FRMT pt-pt',
        'frmt_zeroshot_pt': 'FRMT pt-pt (0-shot)',
        'bbh_cot_fewshot_mt_pt': 'BBH',
        'ifeval-mt-pt': 'IFEval',
        'dsl_tl_pt_mcq_shuffled_no_na_chat': 'DSL',
        'dsl_tl_pt_mcq_shuffled_no_na': 'DSL (base)',
        'ptbrvid_mcq_shuffle_chat': 'VID',
        'ptbrvid_mcq_shuffle': 'VID (base)',
        'legalbench_pt_log': 'LPT',
        'alba_mcq': 'ALBA',
        'alba_two_options_mcq': 'ALBA-e',
        # alba subcategories
        'alba_discourse': 'ALBA Discourse',
        'alba_lexicology': 'ALBA Lexicology',
        'alba_morphology': 'ALBA Morphology',
        'alba_phonetics': 'ALBA Phonetics',
        'alba_semantics': 'ALBA Semantics',
        'alba_syntax': 'ALBA Syntax',
        'alba_variety': 'ALBA Variety',
        'alba_word_play': 'ALBA Word Play',
        "humaneval_instruct_mt_pt": "HumanEval",
        "cultura_viva_pt_chat": "CulturaVivaPT",
        "cultura_viva_pt_chat_cot": "CulturaVivaPT (CoT)",
        "saudade_chat_pt": "Saudade-PT",
        "saudade_chat_cot_pt": "Saudade-PT (CoT)",
        "calame_pt": "CALAME",
        "mmlu_prox_lite_pt": "MMLU Prox Lite (PT)",

        # english versions
        "arc_challenge_chat": "ARC-C (en)",
        "arc_challenge": "ARC-C (en | base)",
        "mmlu": "MMLU (en)",
        'piqa': 'PiQA (en)',
        "commonsense_qa": "CSQA (en)",
        "social_iqa": "SIQA (en)",
        "gsm8k_cot": "GSM8K (en)",
        "gsm8k_cot_llama": "GSM8K (llama - en)",
        'aime': 'AIME (en)',
        'aime24': 'AIME-24 (en)',
        'aime25': 'AIME-25 (en)',
        'minerva_math500_en': 'MATH 500 (Minerva) (en)',
        'minerva_math_en': 'MATH Hendrycks (Minerva) (en)',
        "hellaswag": "HelSw (en)",
        "bbh_cot_fewshot": "BBH (en)",
        "ifeval": "IFEval (en)",
        "openbookqa": "OBQA (en)",
        "truthfulqa_mc2": "TQA (en)",
        "truthfulqa_mc1": "TQA (en | mc1)",
        "winogrande": "WG (en)",
        "humaneval_instruct": "HumanEval (en)",
        "humaneval_instruct_en": "HumanEval (en)",

        # long context tasks
        "babilong": "BabiLong",
        "babilong_longctx": "BabiLong (LC)",
        "babilong_longctx 4k": "BabiLong (LC) 4k",
        "babilong_longctx 8k": "BabiLong (LC) 8k",
        "babilong_longctx 16k": "BabiLong (LC) 16k",
        "babilong_longctx 32k": "BabiLong (LC) 32k",
        #"ruler_en": "RULER (en)",
        "ruler_en 4096": "RULER (en | 4k)",
        "ruler_en 8192": "RULER (en | 8k)",
        "ruler_en 16384": "RULER (en | 16k)",
        "ruler_en 32768": "RULER (en | 32k)",
        #"ruler_pt": "RULER (pt)",
        "ruler_pt 4096": "RULER (pt | 4k)",
        "ruler_pt 8192": "RULER (pt | 8k)",
        "ruler_pt 16384": "RULER (pt | 16k)",
        "ruler_pt 32768": "RULER (pt | 32k)",
        "scrolls_contractnli_en": "SCROLLS ContractNLI (en)",
        "scrolls_govreport_en": "SCROLLS GovReport (en)",
        "scrolls_qasper_en": "SCROLLS Qasper (en)",
        "scrolls_qmsum_en": "SCROLLS QMSum (en)",
        "scrolls_quality_en": "SCROLLS Quality (en)",
        "scrolls_summscreenfd_en": "SCROLLS SummScreenFD (en)",

        # safety datasets
        "multilingual_advbench_pt": "AdvBench",
        "simple_safety_tests_pt": "SST",
        "xstest_pt": "XSTest",
        "harmbench_standard_pt": "HarmBench-S",
        "harmbench_contextual_pt": "HarmBench-C",
        "wild_guard_test_pt": "WildGuard",
        "toxic_chat_pt": "ToxicChat",

        "multilingual_advbench_en": "AdvBench (en)",
        "simple_safety_tests": "SST (en)",
        "xstest": "XSTest (en)",
        "harmbench_standard": "HarmBench-S (en)",
        "harmbench_contextual": "HarmBench-C (en)",
        "wild_guard_test": "WildGuard (en)",
        "toxic_chat": "ToxicChat (en)",

}

model_to_pretty_name_dict = {

    # base models
    "maritaca-ai/sabia-7b": "Sabia-7B",
    "allenai/OLMo-2-1124-7B": "OLMo 2-7B",
    "BSC-LT/salamandra-7b": "Salamandra-7B",
    "utter-project/EuroLLM-9B": "EuroLLM-9B",
    "utter-project/EuroLLM-9B-Instruct-2512": "EuroLLM-9B-2512",
    "meta-llama/Llama-3.1-8B": "LLaMA 3.1-8B",
    "google/gemma-2-9b": "Gemma 2-9B",
    "Qwen/Qwen2.5-7B": "Qwen 2.5-7B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
    "amalia-llm/AMALIA-9B-base-2509": "AMALIA-9B 32k",

    # instruct models
    # open models
    'allenai/OLMo-2-1124-7B-Instruct': 'OLMo 2-7B',
    'BSC-LT/salamandra-7b-instruct': 'Salamandra-7B',
    'utter-project/EuroLLM-9B-Instruct': 'EuroLLM-9B',
    "swiss-ai/Apertus-8B-Instruct-2509": "Apertus-8B",

    # mistral
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    'mistralai/Ministral-8B-Instruct-2410': 'Ministral-8B',

    # llama
    'meta-llama/Llama-3.1-8B-Instruct': 'LLaMA 3.1-8B',
    'PORTULAN/gervasio-8b-portuguese-ptpt-decoder': 'LLaMA-Gervasio-8B',

    # gemma
    'google/gemma-2-9b-it': 'Gemma 2-9B',
    'google/gemma-3-12b-it': 'Gemma 3-12B',
    "google/gemma-4-E2B-it": "Gemma 4-E2B-it",
    "google/gemma-4-E4B-it": "Gemma 4-E4B-it",
    "google/gemma-4-12B-it": "Gemma 4-12B-it",

    # qwen
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen 2.5-7B',
    "Qwen/Qwen3-8B": "Qwen 3-8B",
    "Qwen/Qwen3.5-9B": "Qwen 3.5-9B",

    # amalia versions
    "amalia-llm/AMALIA-9B-50-1225-SFT": "AMALIA-9B-SFT",
    "amalia-llm/AMALIA-9B-50-1225-DPO": "AMALIA-9B-DPO",

}

def turn_task_to_pretty_name(name: str, verbose: bool = False) -> str:
    # This logic only extracts the base name for matching against dataset_to_prety_name_dict keys
    base_name = name.split("|")[0].strip()

    # special case for ruler and babilong_longctx where we want to include the size
    if base_name.startswith("ruler") or base_name.startswith("babilong_longctx"):
        size_part = name.split("|")[-1].strip() if "|" in name else ""
        if size_part.isdigit() or (size_part.endswith("k") and size_part[:-1].isdigit()):
            return dataset_to_pretty_name_dict.get(f"{base_name} {size_part}", f"{base_name} {size_part}")

    thinking_suffix = " (Think)" if "| thinking" in name else ""

    if base_name in dataset_to_pretty_name_dict:
        return dataset_to_pretty_name_dict[base_name] + thinking_suffix
    else:
        if verbose:
            print(f"Warning: Task name '{name}' not found in dataset_to_prety_name_dict. Using original name.")
        return name


# rename model_names to only include the most relevant parts
def turn_model_to_pretty_name(model_name: str) -> str:

    # try to pretty name using the model_to_pretty_name_dict first
    if model_name in model_to_pretty_name_dict:
        return model_to_pretty_name_dict[model_name]

    parts = model_name.split('/')
    if len(parts) >= 2:
        # check if the second last part is "snapshots" get the proper model name
        if "snapshots" in parts[-2].lower():
            # if the second last part is snapshots, include only part
            return parts[-3].replace("models--", "").replace("--", "/")
        else:
            return '/'.join(parts[-2:])
    return model_name