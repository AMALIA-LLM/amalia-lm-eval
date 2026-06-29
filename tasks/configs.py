"""Task-level harness configuration for AMALIA Eval.

HARNESS_TASK_CONFIGS defines per-task defaults for harness parameters. These
values seed the configuration for a given task and are applied before
model-specific overrides and user CLI overrides.

Precedence (from lowest to highest):
1) HarnessArgs dataclass defaults (safe base for required keys)
2) HARNESS_TASK_CONFIGS (task defaults; this file)
3) HARNESS_MODEL_CONFIGS (model overrides; see configs/models.py)
4) CLI overrides (explicit user-provided values)

Only values explicitly set on the CLI override prior layers; leaving a CLI
field unspecified will not reset values coming from task or model configs.
"""

# HARNESS_TASK_CONFIGS for AMALIA Eval
# Each key is a task name, values are task-specific configuration dicts
_DEV_TASK_CONFIGS = {
    # === Dev Tasks ===
    "arc_challenge_chat": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "boolq": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "commonsense_qa": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "hellaswag": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "mmlu": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "openbookqa": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "piqa": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "social_iqa": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    # === Dev Tasks pt-pt ===
    "amalia_arc_challenge_chat_mt_pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "boolq-mt-pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "commonsense_qa-mt-pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "amalia_hellaswag_mt_pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "amalia_mmlu_mt_pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "openbookqa-mt-pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
    },
    "piqa-mt-pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    "social_iqa-mt-pt": {
        "num_fewshot": 5,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "limit": 1000,
    },
    # === Generative Tasks ===
    "coqa": {
        "apply_chat_template": True,
        "num_fewshot": 0,
    },
    "squadv2": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
        "limit": 1000,
    },
    "squadv2_mt_pt": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
        "limit": 1000,
    },
    # === PT Exams ===
    "pt_exams": {
        "apply_chat_template": True,
    },
    "pt_exams_cot": {
        "apply_chat_template": True,
    },
    # === If Eval Tasks ===
    "ifeval": {
        "apply_chat_template": True,
    },
    "ifeval-mt-pt": {
        "apply_chat_template": True,
    },
    # === Held Out Tasks ===
    "bbh_cot_fewshot": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 3,
    },
    "bbh_cot_fewshot_mt_pt": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 3,
    },
    "gsm8k_cot": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 8,
    },
    "amalia_gsm8k_cot_mt_pt": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 8,
    },
    "mmlu_pro": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "amalia_mmlu_prox": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "triviaqa": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "agieval_en": {
        "apply_chat_template": True,
        "num_fewshot": 0,
    },
    # === Other Tasks ===
    "amalia_dsl_tl_2_labels_fewshot": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "amalia_dsl_tl_pt_mcq_shuffled_no_na": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "amalia_frmt_fewshot_br": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "amalia_frmt_fewshot_pt": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "amalia_frmt_zeroshot_br": {
        "apply_chat_template": True,
    },
    "amalia_frmt_zeroshot_pt": {
        "apply_chat_template": True,
    },
    "amalia_legalbench_pt_log": {
        "apply_chat_template": True,
    },
    "amalia_legalbench_pt_cot": {
        "apply_chat_template": True,
    },
    "amalia_pt_text_completion": {
        "apply_chat_template": True,
    },
    "amalia_ptbrvid_mcq_shuffle": {
        "apply_chat_template": True,
        "limit": 1000,
    },
    "super-glue-lm-eval-v1": {
        "apply_chat_template": True,
    },
    "super-glue-lm-eval-v1-mt-pt": {
        "apply_chat_template": True,
    },
    "winogrande": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "winogrande-mt-pt": {
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    },
    "truthfulqa_mc1": {
        "apply_chat_template": True,
    },
    "truthfulqa_mc2": {
        "apply_chat_template": True,
    },
    "amalia_truthfulqa_mc1_mt_pt": {
        "apply_chat_template": True,
    },
}

# TODO:
HARNESS_TASK_CONFIGS = {}

# TODO: to reproduce the evaluation pipeline instead of using these groups
#  please use the run_chat_* scripts instead as those have the exact same parameters while these use default values
TASK_GROUPS = {
    "dev": [
        "amalia_arc_challenge_chat_mt_pt",
        "arc_challenge_chat",
        "amalia_mmlu_mt_pt_chat",
        "mmlu",
        "ifeval-mt-pt",
        "ifeval",
        "amalia_pt_text_completion_chat",
        "pt_exams_chat",
        "pt_exams_cot",
        "commonsense_qa-mt-pt",
        "commonsense_qa",
        "amalia_hellaswag_mt_pt",
        "hellaswag",
        "amalia_frmt_zeroshot_pt",
        "global_piqa_completions_por_latn_port",
        "global_piqa_prompted_por_latn_port",
        "amalia_truthfulqa_mc2_mt_pt",
        "truthfulqa_mc2",
        "amalia_gsm8k_cot_mt_pt",
        "gsm8k_cot",
        "amalia_frmt_fewshot_pt",

    ],
    "held_out": [
        "piqa-mt-pt",
        "amalia_dsl_tl_pt_mcq_shuffled_no_na_chat",
        "amalia_legalbench_pt_log",
        "alba_mcq",
        "alba_two_options_mcq",
        "social_iqa-mt-pt",
        "piqa",
        "openbookqa",
        "openbookqa-mt-pt",
        "winogrande",
        "winogrande-mt-pt",
        "social_iqa",
        "amalia_ptbrvid_mcq_shuffle_chat",
        "bbh_cot_fewshot_mt_pt",
        "bbh_cot_fewshot",
    ],
    "safety": [
        "xstest",
        "xstest_pt",
        "simple_safety_tests",
        "simple_safety_tests_pt",
        "multilingual_advbench_en",
        "multilingual_advbench_pt",
        # new safety datasets
        "harmbench_standard",
        "harmbench_standard_pt",
        "harmbench_contextual",
        "harmbench_contextual_pt",
        "wild_guard_test",
        "wild_guard_test_pt",
        "toxic_chat",
        "toxic_chat_pt",
    ],
}
