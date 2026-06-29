import evaluate as hf_evaluate
import evaluate.config as eval_config
import os
import dotenv

# load environment variables from .env file
dotenv.load_dotenv()

# Isolate only the evaluate metrics cache per SLURM job to avoid
# arrow file collisions when multiple jobs share the same filesystem
slurm_job_id = os.getenv("SLURM_JOB_ID", "local")
process_id = os.getpid()
eval_config.HF_EVALUATE_CACHE = os.path.join(
    eval_config.HF_EVALUATE_CACHE,
    f"job_{slurm_job_id}_{process_id}"
)
os.makedirs(eval_config.HF_EVALUATE_CACHE, exist_ok=True)
print(f"Using per-job evaluate cache: {eval_config.HF_EVALUATE_CACHE}")

code_eval = "code_eval"
code_eval_local_path = os.getenv("CODE_EVAL_FILE_PATH", None)
# check if file exists
if code_eval_local_path and os.path.isfile(code_eval_local_path):
    print("Using local code_eval metric from path:", code_eval_local_path)
    code_eval = code_eval_local_path

try:
    print("Loading code_eval metric from Hugging Face Hub...")
    compute_ = hf_evaluate.load(code_eval)
    print("Finished loading code_eval metric.")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
