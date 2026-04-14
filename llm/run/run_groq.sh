#!/bin/bash

# Configuration and path setup mapping to data_minidev
eval_path='../data_minidev/MINIDEV/mini_dev_sqlite.json' # Depending on SQL Dialect: _mysql.json, _postgresql.json
db_root_path='../data_minidev/MINIDEV/dev_databases/'
use_knowledge='True' #active knowledge base
mode='mini_dev' # dev, train, mini_dev
cot='True' #active Chain-of-Thought made for LLMs

# Parse arguments
num_samples=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_samples) num_samples="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Build the num_samples args if passed
num_samples_arg=""
if [ -n "$num_samples" ]; then
    num_samples_arg="--num_samples $num_samples"
    echo "Info: Running inference on a subset of $num_samples samples."
fi

# Read from Env Variables (loaded via Docker or manually in terminal)
YOUR_API_KEY=${GROQ_API_KEY:-"YOUR_GROQ_API_KEY_HERE"}

# Choose the engine to run, e.g. meta-llama/llama-4-scout-17b-16e-instruct or llama-3.1-70b-versatile
engine=${GROQ_ENGINE:-'meta-llama/llama-4-scout-17b-16e-instruct'}

# Choose the number of threads to run in parallel
num_threads=${NUM_THREADS:-3}

# Choose the SQL dialect to run, e.g. SQLite, MySQL, PostgreSQL
sql_dialect='SQLite'

# Choose the output path for the generated SQL queries
data_output_path='./exp_result/groq_output/'
data_kg_output_path='./exp_result/groq_output_kg/'

echo "Generate $engine batch, run in $num_threads threads, with knowledge: $use_knowledge, with chain of thought: $cot"

# Execute Python script for Groq Llama 4 Scout
python3 -u ./src/groq_request.py \
    --db_root_path "${db_root_path}" \
    --api_key "${YOUR_API_KEY}" \
    --mode "${mode}" \
    --engine "${engine}" \
    --eval_path "${eval_path}" \
    --data_output_path "${data_kg_output_path}" \
    --use_knowledge "${use_knowledge}" \
    --chain_of_thought "${cot}" \
    --num_process "${num_threads}" \
    --sql_dialect "${sql_dialect}" \
    $num_samples_arg
