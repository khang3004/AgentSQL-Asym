"""
Master Pipeline Evaluation Suite
Evaluates the MasterPipeline (CHESS + MCI-SQL + MAGIC) on the mini_dev dataset.
"""
import os
import json
import time
import sqlite3
import argparse
import logging
import math
import numpy as np
from statistics import mean

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from text2sql_agent.tools.master_pipeline import MasterPipeline
from research.evaluator import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def evaluate_master_pipeline(dataset: list, db_root: str, ves_iters: int = 5, top_k: int = 3, force_restart: bool = False):
    logger.info("Starting MasterPipeline Evaluation")
    
    results_file = "results/master_pipeline_evaluation.json"
    os.makedirs("results", exist_ok=True)
    
    results_log = []
    processed_qids = set()
    
    if not force_restart and os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results_log = json.load(f)
            processed_qids = {r.get("question_id") for r in results_log if "question_id" in r}
            logger.info("Resuming from checkpoint: %d samples already evaluated.", len(processed_qids))
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Starting fresh.", e)
            results_log = []
            
    # Calculate initial stats from loaded results
    total_api_calls = sum(r.get("api_calls", 1) for r in results_log)
    latencies = [r.get("latency", 0) for r in results_log]
    correct_ex = sum(1 for r in results_log if r.get("is_correct"))
    total_f1 = sum(r.get("f1_score", 0.0) for r in results_log)
    total_ves_reward = sum(math.sqrt(r.get("ves_reward", 0.0)) * 100 for r in results_log)
    
    # Initialize the pipeline
    pipeline = MasterPipeline(
        top_k=top_k,
        generator_provider=os.getenv("GENERATOR_PROVIDER", "groq"),
        generator_model=os.getenv("GENERATOR_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        critic_provider=os.getenv("CRITIC_PROVIDER", "google"),
        critic_model=os.getenv("CRITIC_MODEL", "gemini-2.5-flash")
    )
    
    for idx, sample in enumerate(dataset):
        q_id = sample.get("question_id", idx)
        if q_id in processed_qids:
            continue
            
        db_id = sample["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        
        start_time = time.time()
        try:
            result = pipeline.run(question=sample["question"], db_path=db_path)
            predicted = result.final_sql
            api_calls = result.api_calls_made
        except Exception as e:
            logger.error(f"Error evaluating sample {idx}: {e}")
            predicted = ""
            api_calls = 1
            
        latency = time.time() - start_time
        latencies.append(latency)
        total_api_calls += api_calls
        
        ground_truth = sample.get("SQL", "")
        
        metrics = calculate_metrics(predicted, ground_truth, db_path, ves_iters=ves_iters)
        correct_ex += metrics["ex"]
        total_f1 += metrics["f1"]
        total_ves_reward += math.sqrt(metrics["ves_reward"]) * 100
            
        results_log.append({
            "question_id": q_id,
            "db_id": db_id,
            "question": sample["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": metrics["ex"] > 0,
            "f1_score": metrics["f1"],
            "ves_reward": metrics["ves_reward"],
            "api_calls": api_calls,
            "latency": latency
        })
        
        # Save checkpoint after each sample
        with open(results_file, "w") as f:
            json.dump(results_log, f, indent=4)
            
    num_samples = len(dataset)
    ex = (correct_ex / num_samples * 100) if num_samples else 0
    ves = (total_ves_reward / num_samples) if num_samples else 0
    soft_f1 = (total_f1 / num_samples * 100) if num_samples else 0
    avg_latency = mean(latencies) if latencies else 0
    
    return ex, ves, soft_f1, total_api_calls, avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K parameter for CHESS pruning")
    parser.add_argument("--force-restart", action="store_true", help="Ignore checkpoint and start evaluation from the beginning")
    args = parser.parse_args()
    
    _DB_ROOT = "../data_minidev/MINIDEV/dev_databases"
    _DATASET_PATH = "../data_minidev/MINIDEV/mini_dev_sqlite.json"
    
    if not os.path.exists(_DB_ROOT):
        _DB_ROOT = "data_minidev/MINIDEV/dev_databases"
        _DATASET_PATH = "data_minidev/MINIDEV/mini_dev_sqlite.json"
    
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if args.num_samples > 0:
            dataset = dataset[:args.num_samples]
        
    print(f"=== Running MasterPipeline Evaluation on {len(dataset)} samples ===")
    if args.force_restart:
        print("Force restart enabled: ignoring previous checkpoints.")
    
    ex, ves, f1, api_calls, latency = evaluate_master_pipeline(dataset, _DB_ROOT, top_k=args.top_k, force_restart=args.force_restart)
    
    print("\n" + "="*50)
    print(f"[MasterPipeline Results]")
    print(f"Execution Accuracy (EX): {ex:.2f}%")
    print(f"Valid Execution Score (VES): {ves:.2f}")
    print(f"Soft F1 Score: {f1:.2f}%")
    print(f"Total API Calls: {api_calls}")
    print(f"Average Latency: {latency:.2f}s per query")
    print("="*50)
