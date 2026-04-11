#!/bin/bash

# Configuration and path setup mapping to data_minidev
eval_path='../data_minidev/MINIDEV/mini_dev_sqlite.json' # Depending on SQL Dialect: _mysql.json, _postgresql.json
db_root_path='../data_minidev/MINIDEV/dev_databases/'
use_knowledge='True'
mode='mini_dev' # dev, train, mini_dev
cot='True'

# Read from Env Variables (loaded via Docker or manually in terminal)
YOUR_API_KEY=${GEMINI_API_KEY:-"YOUR_GEMINI_API_KEY_HERE"}

# Choose the engine to run, e.g. gemini-2.5-flash, gemini-2.5-pro
engine=${GEMINI_ENGINE:-'gemini-2.5-flash'}

# Choose the number of threads to run in parallel
num_threads=${NUM_THREADS:-1}

# Choose the SQL dialect to run, e.g. SQLite, MySQL, PostgreSQL
sql_dialect='SQLite'

# Choose the output path for the generated SQL queries
data_output_path='./exp_result/gemini_output/'
data_kg_output_path='./exp_result/gemini_output_kg/'

echo "Generate $engine batch, run in $num_threads threads, with knowledge: $use_knowledge, with chain of thought: $cot"

# Execute Python script for Gemini
python3 -u ./src/gemini_request.py \
    --db_root_path "${db_root_path}" \
    --api_key "${YOUR_API_KEY}" \
    --mode "${mode}" \
    --engine "${engine}" \
    --eval_path "${eval_path}" \
    --data_output_path "${data_kg_output_path}" \
    --use_knowledge "${use_knowledge}" \
    --chain_of_thought "${cot}" \
    --num_process "${num_threads}" \
    --sql_dialect "${sql_dialect}"
