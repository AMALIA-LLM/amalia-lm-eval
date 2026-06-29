"""Model-level harness configuration for AMALIA Eval.

HARNESS_MODEL_CONFIGS provides per-model overrides for harness parameters. These
are applied after task defaults and before user CLI overrides:

Precedence:
1) HARNESS_TASK_CONFIGS (task defaults)
2) HARNESS_MODEL_CONFIGS (model overrides)
3) CLI overrides (explicit user-provided values)

Note: Dataclass defaults in HarnessArgs are used only as a safe base to ensure
keys exist; they do not override task/model-specific values. Only values the
user explicitly supplies on the CLI will override prior layers.
"""

# Model configuration for AMALIA Eval
# Each key is a HuggingFace model ID, values are kwargs for model loading/running
HARNESS_MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {
        "apply_chat_template": False,
    },
    "utter-project/EuroLLM-9B": {
        "apply_chat_template": False,
    },
    "allenai/OLMo-2-1124-7B": {
        "apply_chat_template": False,
    },
    "BSC-LT/salamandra-7b": {
        "apply_chat_template": False,
    },
    "amalia-llm/AMALIA-9B-base-2509": {
        "apply_chat_template": False,
    },
    "maritaca-ai/sabia-7b": {
        "apply_chat_template": False,
        "num_fewshot": 1,  # due to limitations of the model context length (was 2 for OQA, and 5 for the rest)
        "model_args": {"max_model_len": "2048"},
    },
    "utter-project/EuroLLM-1.7B": {
        "apply_chat_template": False
    },
    "Qwen/Qwen2.5-1.5B": {
        "apply_chat_template": False
    },
    "allenai/OLMo-2-0425-1B": {
        "apply_chat_template": False
    },
    "mistralai/Mistral-7B-v0.3": {
        "apply_chat_template": False
    },
}
