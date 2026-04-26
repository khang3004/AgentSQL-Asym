import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from text2sql_agent.workflow.graph import compile_workflow

def main():
    graph = compile_workflow()
    
    # Path relative to inside container where docker executes it
    dev_db_path = "/app/data_minidev/MINIDEV/dev_databases/debit_card_specializing/debit_card_specializing.sqlite"
    
    if not os.path.exists(dev_db_path):
        import sys
        sys.exit(f"Cannot find db: {dev_db_path}")

    # GT SQL from bird dataset for question 1471
    gt_sql = "SELECT CAST(SUM(CASE WHEN Currency = 'EUR' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN Currency = 'CZK' THEN 1 ELSE 0 END) FROM customers"
    
    initial_state = {
        "question": "What is the ratio of customers who pay in EUR against customers who pay in CZK?",
        "db_path": dev_db_path,
        "schema_context": "",
        "generated_sql": "",
        "ground_truth_sql": gt_sql,
        "execution_feedback": "",
        "guideline": "",
        "iteration_count": 0
    }
    
    logging.info("Starting graph invoke...")
    final_state = graph.invoke(initial_state)
    
    print("\n--- SMOKE TEST FINAL STATE ---")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    main()
