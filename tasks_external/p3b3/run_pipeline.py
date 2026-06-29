#!/usr/bin/env python3

import argparse
import subprocess
import sys
import sqlite3
import pandas as pd
import time
from pathlib import Path

def run_generation(model_name, output_dir, system_prompt=""):
    """Run generate.py to generate responses."""
    print(f"\n{'='*60}")
    print(f"🚀 STEP 1/3: Running generation for {model_name}")
    print(f"{'='*60}")
    start_time = time.time()
    
    cmd = [
        "python", "generate.py",
        "--model-name-or-path", model_name,
        "--base-output-dir", output_dir
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Generation failed after {elapsed:.1f}s")
        return False
    print(f"✅ Generation completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    return True

def run_class_scoring(base_dir):
    """Run score.py for classifier scoring."""
    print(f"\n{'='*60}")
    print(f"🔍 STEP 2/3: Running classifier scoring on {base_dir}")
    print(f"{'='*60}")
    start_time = time.time()
    
    cmd = ["python", "score.py", base_dir]
    
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Classifier scoring failed after {elapsed:.1f}s")
        return False
    print(f"✅ Classifier scoring completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    return True

def run_llm_scoring(base_dir):
    """Run llm_score.py for LLM scoring."""
    print(f"\n{'='*60}")
    print(f"🤖 STEP 3/3: Running LLM scoring on {base_dir}")
    print(f"{'='*60}")
    start_time = time.time()
    
    cmd = ["python", "llm_score.py", base_dir]
    
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ LLM scoring failed after {elapsed:.1f}s")
        return False
    print(f"✅ LLM scoring completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    return True

def export_llm_csvs(base_dir):
    """Export CSVs from LLM database (standalone function)."""
    print(f"\n{'='*60}")
    print("📊 Exporting LLM CSVs from database")
    print(f"{'='*60}")
    start_time = time.time()
    
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"❌ Directory '{base_dir}' does not exist")
        return False
    
    llm_scores_dir = base_path / "llm_scores"
    db_path = llm_scores_dir / "pt_pt_conversation_evaluations.db"
    
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return False
    
    llm_scores_dir.mkdir(exist_ok=True)
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Query all data
        df = pd.read_sql_query("""
            SELECT model_name, conversation_id as prompt_id, turn_number as turn_nr, 
                   prompt_type, score as llm_score, reasoning as llm_reasoning
            FROM evaluations
            ORDER BY model_name, prompt_type, conversation_id, turn_number
        """, conn)
        conn.close()
        
        if df.empty:
            print("⚠️ No data found in database")
            return True
        
        # Group by model_name and prompt_type to create separate CSVs
        for (model_name, prompt_type), group in df.groupby(['model_name', 'prompt_type']):
            suffix = f"_{prompt_type}" if prompt_type != "normal" else ""
            filename = f"{model_name.lower().replace(' ', '-').replace('/', '_')}{suffix}_llm_scored.csv"
            output_path = llm_scores_dir / filename
            
            group.set_index('prompt_id', inplace=True)
            group.to_csv(output_path)
            print(f"📄 Exported {output_path}")
        
        print(f"✅ LLM CSV export completed in {time.time() - start_time:.1f}s")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run P3B3 evaluation pipeline")
    
    # Pipeline steps
    parser.add_argument("--generate", action="store_true", help="Run generation")
    parser.add_argument("--class-score", action="store_true", help="Run classifier scoring")
    parser.add_argument("--llm-score", action="store_true", help="Run LLM scoring")
    parser.add_argument("--export-llm", action="store_true", help="Export LLM CSVs from database (standalone)")
    
    # Generation parameters
    parser.add_argument("--model-name-or-path", type=str, help="Model name or path for generation")
    parser.add_argument("--output-dir", type=str, help="Output directory for generation")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt for generation")
    
    # Scoring parameters
    parser.add_argument("--base-dir", type=str, help="Base directory for scoring (where JSONs are)")
    
    args = parser.parse_args()
    
    pipeline_start = time.time()
    
    # Check if export-llm is used alone
    if args.export_llm:
        if any([args.generate, args.class_score, args.llm_score]):
            print("❌ --export-llm must be used alone")
            sys.exit(1)
        if not args.base_dir:
            print("❌ --base-dir required for LLM export")
            sys.exit(1)
        success = export_llm_csvs(args.base_dir)
        sys.exit(0 if success else 1)
    
    # Check if any pipeline step is selected
    if not any([args.generate, args.class_score, args.llm_score]):
        print("❌ Please select at least one pipeline step")
        parser.print_help()
        sys.exit(1)
    
    # Count total steps
    total_steps = sum([args.generate, args.class_score, args.llm_score])
    print(f"\n{'='*60}")
    print(f"🎯 Starting pipeline with {total_steps} step(s)")
    print(f"{'='*60}")
    
    success = True
    current_step = 0
    
    # Run generation
    if args.generate:
        current_step += 1
        if not args.model_name_or_path or not args.output_dir:
            print("❌ --model-name-or-path and --output-dir required for generation")
            sys.exit(1)
        success &= run_generation(args.model_name_or_path, args.output_dir, args.system_prompt)
    
    # Run classifier scoring
    if args.class_score:
        current_step += 1
        base_dir = args.base_dir or args.output_dir
        if not base_dir:
            print("❌ --base-dir required for classifier scoring")
            sys.exit(1)
        success &= run_class_scoring(base_dir)
    
    # Run LLM scoring
    if args.llm_score:
        current_step += 1
        base_dir = args.base_dir or args.output_dir
        if not base_dir:
            print("❌ --base-dir required for LLM scoring")
            sys.exit(1)
        success &= run_llm_scoring(base_dir)
    
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    if success:
        print(f"🎉 Pipeline completed successfully!")
        print(f"⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    else:
        print(f"💥 Pipeline failed after {total_elapsed:.1f}s")
        sys.exit(1)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()