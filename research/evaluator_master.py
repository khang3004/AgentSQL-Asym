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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.src.text2sql_agent.tools.master_pipeline import MasterPipeline
from research.evaluator import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def evaluate_master_pipeline(dataset: list, db_root: str, ves_iters: int = 5, top_k: int = 3):
    logger.info("Starting MasterPipeline Evaluation")
    
    # Initialize the pipeline
    pipeline = MasterPipeline(
        top_k=top_k,
        generator_provider=os.getenv("GENERATOR_PROVIDER", "groq"),
        generator_model=os.getenv("GENERATOR_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        critic_provider=os.getenv("CRITIC_PROVIDER", "google"),
        critic_model=os.getenv("CRITIC_MODEL", "gemini-2.5-flash")
    )
    
    total_api_calls = 0
    latencies = []
    correct_ex = 0
    total_f1 = 0
    total_ves_reward = 0
    results_log = []
    
    for idx, sample in enumerate(dataset):
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
            "question_id": sample.get("question_id", idx),
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
            
    num_samples = len(dataset)
    ex = (correct_ex / num_samples * 100) if num_samples else 0
    ves = (total_ves_reward / num_samples) if num_samples else 0
    soft_f1 = (total_f1 / num_samples * 100) if num_samples else 0
    avg_latency = mean(latencies) if latencies else 0
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open("results/master_pipeline_evaluation.json", "w") as f:
        json.dump(results_log, f, indent=4)
        
    return ex, ves, soft_f1, total_api_calls, avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K parameter for CHESS pruning")
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
    
    ex, ves, f1, api_calls, latency = evaluate_master_pipeline(dataset, _DB_ROOT, top_k=args.top_k)
    
    print("\n" + "="*50)
    print(f"[MasterPipeline Results]")
    print(f"Execution Accuracy (EX): {ex:.2f}%")
    print(f"Valid Execution Score (VES): {ves:.2f}")
    print(f"Soft F1 Score: {f1:.2f}%")
    print(f"Total API Calls: {api_calls}")
    print(f"Average Latency: {latency:.2f}s per query")
    print("="*50)
