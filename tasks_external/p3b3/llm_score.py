import re
import time
import json
import sqlite3
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path
from threading import Lock
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Model, Gemini
from prompts import PROMPT_PT_PT_EVAL
 
DEBUG = False
MAX_WORKERS = 10
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

# Global variables (will be set in main)
LLM_SCORES_DIR = None
DB_NAME = None

client: Model = Gemini('gemini-2.5-pro')
db_lock = Lock()
evaluated_cache = set()

def load_evaluated_cache() -> Set[Tuple]:
    """Load all evaluated entries into memory for fast lookup."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT model_name, conversation_id, turn_number, prompt_type FROM evaluations")
    cache: Set[Tuple] = set(cursor.fetchall())
    conn.close()
    return cache

def evaluate_turn(model_name, conversation_id, turn_number, context, response, prompt_type):
    # Check cache instead of querying database
    if (model_name, conversation_id, turn_number, prompt_type) in evaluated_cache:
        return None
    
    eval_prompt: str = PROMPT_PT_PT_EVAL.format(prompt=context, response=response)
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            result = client.chat(eval_prompt)
            # Extract JSON from response (because we asked for a json format response)
            json_match = re.search(r'```json\s*({.*?})\s*```', result, re.DOTALL)
            if json_match:
                result = json_match.group(1).strip()
            else:
                json_match = re.search(r'({\s*".*?})', result, re.DOTALL)
                if json_match:
                    result = json_match.group(1).strip()
            
            result_json = json.loads(result)
            score = int(result_json["score"])
            reasoning = result_json["reasoning"]
            
            # Put in the database
            with db_lock:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO evaluations (model_name, conversation_id, turn_number, context, response, score, reasoning, prompt_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (model_name, conversation_id, turn_number, context, response, score, reasoning, prompt_type))
                conn.commit()
                conn.close()
            
            return {"success": True}
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                if DEBUG:
                    print(f"Failed: {model_name} - {conversation_id} - turn {turn_number}: {e}")
                return {"success": False}

def process_conversation(conversation_list: List[Dict], model_name: str, conversation_id: str, prompt_type: str) -> List[Tuple]:
    """
    Process a conversation list and create evaluation tasks.
    Each turn is evaluated with cumulative context.
    """
    tasks = []
    
    for turn_idx, turn in enumerate(conversation_list, start=1):
        # Build cumulative context: all previous turns + current user message
        context_parts = []
        for prev_turn in conversation_list[:turn_idx-1]:
            context_parts.append(f"User: {prev_turn['user']}")
            context_parts.append(f"Assistant: {prev_turn['assistant']}")
        context_parts.append(f"User: {turn['user']}")
        
        context = "\n".join(context_parts)
        response = turn['assistant']
        
        tasks.append((model_name, conversation_id, turn_idx, context, response, prompt_type))
    
    return tasks

def load_conversations(base_dir: Path) -> List[Tuple]:
    """
    Load conversations from specified directory.
    Returns list of (model_name, conversation_id, conversation_list, prompt_type) tuples.
    """
    conversations = []

    for json_file in base_dir.glob("*.json"):
        # Determine prompt type from filename
        filename = json_file.stem
        if filename.endswith("_pt-pt"):
            prompt_type = "pt-pt"
        elif filename.endswith("_pt-br"):
            prompt_type = "pt-br"
        else:
            prompt_type = "normal"

        # Load JSON data
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            if DEBUG:
                print(f"Failed to load {json_file}: {e}")
            continue

        # Extract model name from JSON
        if not data or len(data) == 0:
            if DEBUG:
                print(f"Empty data in {json_file}")
            continue
        
        model_name = data[0].get("model_name", "")
        if not model_name:
            if DEBUG:
                print(f"No model_name in {json_file}")
            continue

        # Process each conversation in the file
        for conv in data:
            conv: Dict = conv
            conversation_id: str = conv.get("prompt_id")
            conversation_list: List[Dict] = conv.get("turns")
            # Skip malformed entries
            if conversation_id is None or conversation_list is None:
                if DEBUG:
                    print(f"Skipping malformed convo in {json_file}: {conv}")
                continue
            conversations.append((model_name, conversation_id, conversation_list, prompt_type))

    return conversations

def process_all_conversations(conversations: List[Tuple]) -> List[Tuple]:
    """
    Process all conversations and create evaluation tasks.
    
    Args:
        conversations: List of (model_name, conversation_id, conversation_list, prompt_type) tuples
    
    Returns:
        List of evaluation tasks as tuples: (model_name, conversation_id, turn_idx, context, response, prompt_type)
    """
    all_tasks = []
    for model_name, conversation_id, conversation_list, prompt_type in conversations:
        tasks: List[Tuple] = process_conversation(conversation_list, model_name, conversation_id, prompt_type)
        all_tasks.extend(tasks)
    return all_tasks

def run_evaluations(all_tasks):
    """Run parallel evaluations on all tasks."""
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_turn, *task): task for task in all_tasks}
        
        with tqdm(total=len(all_tasks), desc="Evaluating") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                # TQDM update
                if result is None:
                    pbar.set_postfix({"skipped": "already evaluated"})
                elif result["success"]:
                    success_count += 1
                    pbar.set_postfix({"success": success_count, "failed": failed_count})
                else:
                    failed_count += 1
                    pbar.set_postfix({"success": success_count, "failed": failed_count})
                pbar.update(1)
    
    return success_count, failed_count

def export_csvs():
    """Export CSV files from database similar to class_scores format."""
    conn = sqlite3.connect(DB_NAME)
    
    # Query all data
    df = pd.read_sql_query("""
        SELECT model_name, conversation_id as prompt_id, turn_number as turn_nr, 
               prompt_type, score as llm_score, reasoning as llm_reasoning
        FROM evaluations
        ORDER BY model_name, prompt_type, conversation_id, turn_number
    """, conn)
    conn.close()
    
    # Group by model_name and prompt_type to create separate CSVs
    for (model_name, prompt_type), group in df.groupby(['model_name', 'prompt_type']):
        suffix = f"_{prompt_type}" if prompt_type != "normal" else ""
        filename = f"{model_name.lower().replace(' ', '-').replace('/', '_')}{suffix}_llm_scored.csv"
        output_path = LLM_SCORES_DIR / filename
        
        group.set_index('prompt_id', inplace=True)
        group.to_csv(output_path)
        print(f"Exported {output_path}")

def main():
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <base-dir>', file=sys.stderr)
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.is_dir():
        print(f"Error: Directory '{base_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create llm_scores folder and set DB path
    global LLM_SCORES_DIR, DB_NAME
    LLM_SCORES_DIR = base_dir / "llm_scores"
    LLM_SCORES_DIR.mkdir(exist_ok=True)
    DB_NAME = str(LLM_SCORES_DIR / "pt_pt_conversation_evaluations.db")
    
    # Initialize database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            conversation_id TEXT,
            turn_number INTEGER,
            context TEXT,
            response TEXT,
            score INTEGER,
            reasoning TEXT,
            prompt_type TEXT DEFAULT 'normal'
        )
    """)
    conn.commit()
    conn.close()

    # Load evaluated cache for fast lookup: Set of Tuple -> (model_name, conversation_id, turn_number, prompt_type)
    print("Loading evaluated cache...")
    global evaluated_cache
    evaluated_cache = load_evaluated_cache() # -> This is a global variable that will be used in evaluate_turn
    print(f"Loaded {len(evaluated_cache)} already evaluated entries")

    # Load conversations from data source: List of Tuple -> (model_name, conversation_id, conversation_list, prompt_type)
    conversations: List[Tuple] = load_conversations(base_dir)
    print(f"Loaded {len(conversations)} conversations")

    # Process all conversations into evaluation tasks: List of Tuple -> (model_name, conversation_id, turn_idx, context, response, prompt_type)
    all_tasks: List[Tuple] = process_all_conversations(conversations)
    print(f"Total turns to evaluate: {len(all_tasks)}")

    # Run evaluations
    success_count, failed_count = run_evaluations(all_tasks)
    print(f"\nCompleted: {success_count} successful, {failed_count} failed")
    
    # Export CSVs
    print("\nExporting CSVs...")
    export_csvs()
    print("Done!")

if __name__ == "__main__":
    main()
