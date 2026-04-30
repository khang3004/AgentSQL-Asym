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
import math
import numpy as np
from statistics import mean

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.src.text2sql_agent.workflow.graph import compile_workflow
from llm.src.text2sql_agent.core.state import AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def calculate_row_match(predicted_row, ground_truth_row):
    if not ground_truth_row:
        return 0, 0, 0
    total_columns = len(ground_truth_row)
    matches = 0
    element_in_pred_only = 0
    element_in_truth_only = 0
    for pred_val in predicted_row:
        if pred_val in ground_truth_row:
            matches += 1
        else:
            element_in_pred_only += 1
    for truth_val in ground_truth_row:
        if truth_val not in predicted_row:
            element_in_truth_only += 1
    match_percentage = matches / total_columns
    pred_only_percentage = element_in_pred_only / total_columns
    truth_only_percentage = element_in_truth_only / total_columns
    return match_percentage, pred_only_percentage, truth_only_percentage

def calculate_f1_score(predicted, ground_truth):
    if not predicted and not ground_truth:
        return 1.0
    if not ground_truth:
        return 0.0
    
    predicted = list(dict.fromkeys(predicted))
    ground_truth = list(dict.fromkeys(ground_truth))

    match_scores = []
    pred_only_scores = []
    truth_only_scores = []
    for i, gt_row in enumerate(ground_truth):
        if i >= len(predicted):
            match_scores.append(0)
            truth_only_scores.append(1)
            continue
        pred_row = predicted[i]
        match_score, pred_only_score, truth_only_score = calculate_row_match(pred_row, gt_row)
        match_scores.append(match_score)
        pred_only_scores.append(pred_only_score)
        truth_only_scores.append(truth_only_score)

    for i in range(len(predicted) - len(ground_truth)):
        match_scores.append(0)
        pred_only_scores.append(1)
        truth_only_scores.append(0)

    tp = sum(match_scores)
    fp = sum(pred_only_scores)
    fn = sum(truth_only_scores)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)
    return f1_score

def clean_abnormal(input_list):
    if not input_list: return []
    input_arr = np.asarray(input_list)
    mean_val = np.mean(input_arr)
    std_val = np.std(input_arr)
    return [x for x in input_list if x < mean_val + 3 * std_val and x > mean_val - 3 * std_val]

def calculate_metrics(sql1: str, sql2: str, db_path: str, ves_iters: int = 5) -> dict:
    metrics = {"ex": 0.0, "f1": 0.0, "ves_reward": 0.0}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Predicted results
        cursor.execute(sql1)
        res1 = cursor.fetchall()
        
        # Ground truth results
        cursor.execute(sql2)
        res2 = cursor.fetchall()
        
        # EX (Execution Accuracy)
        if set(res1) == set(res2):
            metrics["ex"] = 1.0
            
            # VES Reward (Timing based)
            diff_list = []
            for _ in range(ves_iters):
                start = time.time()
                cursor.execute(sql1)
                cursor.fetchall()
                t1 = time.time() - start
                
                start = time.time()
                cursor.execute(sql2)
                cursor.fetchall()
                t2 = time.time() - start
                
                if t1 == 0: t1 = 1e-9
                diff_list.append(t2 / t1)
            
            processed_diff = clean_abnormal(diff_list)
            if processed_diff:
                time_ratio = sum(processed_diff) / len(processed_diff)
                if time_ratio >= 2: metrics["ves_reward"] = 1.25
                elif time_ratio >= 1: metrics["ves_reward"] = 1.0
                elif time_ratio >= 0.5: metrics["ves_reward"] = 0.75
                elif time_ratio >= 0.25: metrics["ves_reward"] = 0.5
                else: metrics["ves_reward"] = 0.25
        
        # Soft F1 Score
        metrics["f1"] = calculate_f1_score(res1, res2)
        
        conn.close()
        return metrics
    except Exception:
        return metrics

def evaluate_agent_sql(dataset: list, db_root: str, ves_iters: int = 5):
    logger.info("Starting AgentSQL Evaluation")
    graph = compile_workflow()
    
    total_tokens = 0
    latencies = []
    correct_ex = 0
    total_f1 = 0
    total_ves_reward = 0
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
            "ground_truth_sql": sample.get("SQL", ""),
            "evidence": sample.get("evidence", ""),
            "difficulty": sample.get("difficulty", ""),
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
            "iterations": iters,
            "latency": latency
        })
            
    num_samples = len(dataset)
    ex = (correct_ex / num_samples * 100) if num_samples else 0
    ves = (total_ves_reward / num_samples) if num_samples else 0
    soft_f1 = (total_f1 / num_samples * 100) if num_samples else 0
    avg_latency = mean(latencies) if latencies else 0
    
    # Save detailed results
    os.makedirs("results", exist_ok=True)
    with open("results/agentsql_evaluation.json", "w") as f:
        json.dump(results_log, f, indent=4)
        
    return ex, ves, soft_f1, total_tokens, avg_latency

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
    
    ex, ves, f1, tokens, latency = evaluate_agent_sql(dataset, _DB_ROOT)
    
    print("\n" + "="*50)
    print(f"[AgentSQL Results]")
    print(f"Execution Accuracy (EX): {ex:.2f}%")
    print(f"Valid Execution Score (VES): {ves:.2f}")
    print(f"Soft F1 Score: {f1:.2f}%")
    print(f"Total Token Usage (Est.): {tokens}")
    print(f"Average Latency: {latency:.2f}s per query")
    print("="*50)
