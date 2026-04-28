"""
AgentSQL Evaluation Suite
Evaluates the LangGraph asymmetric system (AgentSQL) on the mini_dev dataset.
"""
import os
import json
import time
import sqlite3
import argparse
import logging
from statistics import mean

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.src.graph_orchestrator import compile_graph, AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def compare_results(sql1: str, sql2: str, db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql1)
        res1 = set(cursor.fetchall())
        cursor.execute(sql2)
        res2 = set(cursor.fetchall())
        conn.close()
        return res1 == res2
    except Exception:
        return False

def evaluate_agent_sql(dataset: list, db_root: str):
    logger.info("Starting AgentSQL Evaluation")
    graph = compile_graph()
    
    total_tokens = 0
    latencies = []
    correct = 0
    results_log = []
    
    for idx, sample in enumerate(dataset):
        db_id = sample["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        
        initial_state: AgentState = {
            "question": sample["question"],
            "db_path": db_path,
            "schema_context": "",
            "generated_sql": "",
            "execution_feedback": "",
            "guideline": "",
            "iteration_count": 0,
        }
        
        start_time = time.time()
        final_state = graph.invoke(initial_state)
        latency = time.time() - start_time
        latencies.append(latency)
        
        iters = final_state.get("iteration_count", 0)
        # Rough token estimation for logging purposes
        total_tokens += 1000 * (1 + 2 * iters) 
        
        ground_truth = sample.get("SQL", "")
        predicted = final_state.get("generated_sql", "")
        
        is_correct = compare_results(predicted, ground_truth, db_path)
        if is_correct:
            correct += 1
            
        results_log.append({
            "question_id": sample.get("question_id", idx),
            "db_id": db_id,
            "question": sample["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "iterations": iters,
            "latency": latency
        })
            
    ex = correct / len(dataset) * 100 if dataset else 0
    avg_latency = mean(latencies) if latencies else 0
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open("results/agentsql_evaluation.json", "w") as f:
        json.dump(results_log, f, indent=4)
        
    return ex, total_tokens, avg_latency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate (0 for all)")
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
        
    print(f"=== Running AgentSQL Evaluation on {len(dataset)} samples ===")
    
    ex, tokens, latency = evaluate_agent_sql(dataset, _DB_ROOT)
    
    print("\n" + "="*50)
    print(f"[AgentSQL Results]")
    print(f"Execution Accuracy (EX): {ex:.2f}%")
    print(f"Total Token Usage (Est.): {tokens}")
    print(f"Average Latency: {latency:.2f}s per query")
    print("="*50)
