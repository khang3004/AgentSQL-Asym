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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from text2sql_agent.workflow.agent_workflow import compile_workflow
from text2sql_agent.core.state import AgentState

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

def calculate_summary_metrics(results_log: list) -> tuple[dict, dict]:
    """Computes overall and difficulty-level metrics from results log."""
    if not results_log:
        return {}, {}
        
    num_samples = len(results_log)
    correct_ex = sum(1 for r in results_log if r.get("is_correct", False))
    ex_accuracy = (correct_ex / num_samples * 100)
    
    total_f1 = sum(r.get("f1_score", 0.0) for r in results_log)
    soft_f1_score = (total_f1 / num_samples * 100)
    
    total_ves_reward = sum(math.sqrt(r.get("ves_reward", 0.0)) * 100 for r in results_log)
    valid_execution_score = (total_ves_reward / num_samples)
    
    total_latency = sum(r.get("latency", 0.0) for r in results_log)
    average_latency = (total_latency / num_samples)
    
    total_token_usage = sum(r.get("token_usage", r.get("iterations", 0) * 2000 + 1000) for r in results_log)
    
    overall_metrics = {
        "ex_accuracy": round(ex_accuracy, 2),
        "valid_execution_score": round(valid_execution_score, 2),
        "soft_f1_score": round(soft_f1_score, 2),
        "average_latency": round(average_latency, 2),
        "total_token_usage": total_token_usage,
        "count": num_samples
    }
    
    # Calculate difficulty-level metrics
    by_difficulty = {}
    for r in results_log:
        diff = r.get("difficulty", "unknown")
        if not diff:
            diff = "unknown"
        by_difficulty.setdefault(diff, []).append(r)
        
    difficulty_metrics = {}
    for diff, items in by_difficulty.items():
        n = len(items)
        c_ex = sum(1 for r in items if r.get("is_correct", False))
        ex_acc = (c_ex / n * 100)
        
        t_f1 = sum(r.get("f1_score", 0.0) for r in items)
        s_f1 = (t_f1 / n * 100)
        
        t_ves = sum(math.sqrt(r.get("ves_reward", 0.0)) * 100 for r in items)
        ves_score = (t_ves / n)
        
        t_lat = sum(r.get("latency", 0.0) for r in items)
        avg_lat = (t_lat / n)
        
        difficulty_metrics[diff] = {
            "ex_accuracy": round(ex_acc, 2),
            "valid_execution_score": round(ves_score, 2),
            "soft_f1_score": round(s_f1, 2),
            "average_latency": round(avg_lat, 2),
            "count": n
        }
        
    return overall_metrics, difficulty_metrics

def save_results(results_log: list, filename: str = "results/benchmark_eval_results.json"):
    """Computes metrics and saves the unified results dynamically to the checkpoint file."""
    overall_metrics, difficulty_metrics = calculate_summary_metrics(results_log)
    output_data = {
        "overall_metrics": overall_metrics,
        "difficulty_metrics": difficulty_metrics,
        "results": results_log
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_filename = filename + ".tmp"
    with open(tmp_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    os.replace(tmp_filename, filename)

def evaluate_agent_sql(dataset: list, db_root: str, ves_iters: int = 5):
    logger.info("Starting AgentSQL Evaluation")
    
    # Setup paths and checkpoint file
    results_path = "results/benchmark_eval_results.json"
    existing_results = []
    
    # 1. Load progressive checkpoints
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing_results = data
                    logger.info(f"Loaded {len(existing_results)} existing results from flat list checkpoint.")
                elif isinstance(data, dict):
                    existing_results = data.get("results", [])
                    logger.info(f"Loaded {len(existing_results)} existing results from structured checkpoint.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint from {results_path}: {e}")
            
    # 2. Migrate existing checkpoints to include difficulty
    dataset_by_id = {s["question_id"]: s for s in dataset if "question_id" in s}
    dataset_by_question = {s["question"]: s for s in dataset if "question" in s}
    
    migrated_count = 0
    for r in existing_results:
        if "difficulty" not in r or not r["difficulty"]:
            q_id = r.get("question_id")
            q_text = r.get("question")
            matched_sample = None
            if q_id is not None and q_id in dataset_by_id:
                matched_sample = dataset_by_id[q_id]
            elif q_text is not None and q_text in dataset_by_question:
                matched_sample = dataset_by_question[q_text]
                
            if matched_sample:
                r["difficulty"] = matched_sample.get("difficulty", "unknown")
                migrated_count += 1
            else:
                r["difficulty"] = "unknown"
                
    if migrated_count > 0:
        logger.info(f"Successfully migrated {migrated_count} checkpoint runs to include difficulty field.")
        
    results_log = existing_results
    
    # 3. Track evaluated questions to avoid duplicate runs
    already_evaluated_ids = {r["question_id"] for r in results_log if "question_id" in r}
    already_evaluated_questions = {r["question"] for r in results_log if "question" in r}
    
    # Compile Graph
    graph = compile_workflow()
    
    for idx, sample in enumerate(dataset):
        q_id = sample.get("question_id", idx)
        q_text = sample["question"]
        
        # Check if already processed
        if q_id in already_evaluated_ids or q_text in already_evaluated_questions:
            logger.info(f"Skipping Sample [{idx+1}/{len(dataset)}] (ID: {q_id}) - Already Evaluated.")
            continue
            
        logger.info(f"Evaluating Sample [{idx+1}/{len(dataset)}] (ID: {q_id})")
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
        
        iters = final_state.get("iteration_count", 0)
        token_est = 1000 * (1 + 2 * iters) 
        
        ground_truth = sample.get("SQL", "")
        predicted = final_state.get("generated_sql", "")
        
        metrics = calculate_metrics(predicted, ground_truth, db_path, ves_iters=ves_iters)
        
        # Record this run
        run_result = {
            "question_id": q_id,
            "db_id": db_id,
            "question": sample["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": metrics["ex"] > 0,
            "f1_score": metrics["f1"],
            "ves_reward": metrics["ves_reward"],
            "iterations": iters,
            "latency": latency,
            "difficulty": sample.get("difficulty", "unknown"),
            "token_usage": token_est
        }
        
        results_log.append(run_result)
        
        # Progressive save immediately after each query
        save_results(results_log, filename=results_path)
        logger.info(f"Progressively saved result for question {q_id} to checkpoint.")

    # Calculate final metrics across all results
    overall_metrics, difficulty_metrics = calculate_summary_metrics(results_log)
    return overall_metrics, difficulty_metrics

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
    
    overall, difficulty = evaluate_agent_sql(dataset, _DB_ROOT)
    
    print("\n" + "="*60)
    print(f"               [AgentSQL Evaluation Results]")
    print("="*60)
    print(f"Overall Metrics:")
    print(f"  Execution Accuracy (EX)      : {overall.get('ex_accuracy', 0.0):.2f}%")
    print(f"  Valid Execution Score (VES)  : {overall.get('valid_execution_score', 0.0):.2f}")
    print(f"  Soft F1 Score                : {overall.get('soft_f1_score', 0.0):.2f}%")
    print(f"  Average Latency              : {overall.get('average_latency', 0.0):.2f}s per query")
    print(f"  Total Token Usage (Est.)     : {overall.get('total_token_usage', 0)}")
    print(f"  Total Evaluated Samples      : {overall.get('count', 0)}")
    print("-"*60)
    print("Metrics by Difficulty:")
    for diff, m in sorted(difficulty.items()):
        print(f"  [{diff.upper()}] (Count: {m.get('count', 0)}):")
        print(f"    Execution Accuracy (EX)    : {m.get('ex_accuracy', 0.0):.2f}%")
        print(f"    Valid Execution Score (VES): {m.get('valid_execution_score', 0.0):.2f}")
        print(f"    Soft F1 Score              : {m.get('soft_f1_score', 0.0):.2f}%")
        print(f"    Average Latency            : {m.get('average_latency', 0.0):.2f}s")
    print("="*60)
