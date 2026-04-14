# DO NOT CHANGE THIS
db_root_path='../data_minidev/MINIDEV/dev_databases/'
num_cpus=16
meta_time_out=30.0
# DO NOT CHANGE THIS

# Parse arguments
metrics="ex"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --metrics) metrics="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ************************* #
predicted_sql_path='../llm/exp_result/groq_output_kg/predict_mini_dev_meta-llama_llama-4-scout-17b-16e-instruct_cot_SQLite.json' # Default output path
sql_dialect="SQLite" # ONLY Modify this
# ************************* #

# DO NOT CHANGE THIS
# Extract the base filename without extension
base_name=$(basename "$predicted_sql_path" .json)
# Define the output log path
output_log_path="../eval_result/${base_name}.txt"

# Ensure eval_result directory exists
mkdir -p "../eval_result"

case $sql_dialect in
  "SQLite")
    diff_json_path="../data_minidev/MINIDEV/mini_dev_sqlite.json"
    ground_truth_path="../data_minidev/MINIDEV/mini_dev_sqlite_gold.sql"
    ;;
  *)
    echo "Invalid SQL dialect: $sql_dialect"
    exit 1
    ;;
esac
# DO NOT CHANGE THIS

echo "Differential JSON Path: $diff_json_path"
echo "Ground Truth Path: $ground_truth_path"
echo "Evaluating Metrics: $metrics"

IFS=',' read -ra METRIC_ARRAY <<< "$metrics"
for metric in "${METRIC_ARRAY[@]}"; do
    if [ "$metric" == "ex" ]; then
        echo "========================================="
        echo "starting to compare with knowledge for ex, sql_dialect: ${sql_dialect}"
        python3 -u ./evaluation_ex.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
        --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --output_log_path ${output_log_path} \
        --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --sql_dialect ${sql_dialect}
    fi

    if [ "$metric" == "ves" ]; then
        echo "========================================="
        echo "starting to compare with knowledge for R-VES, sql_dialect: ${sql_dialect}"
        python3 -u ./evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
        --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus}  --output_log_path ${output_log_path} \
        --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}  --sql_dialect ${sql_dialect}
    fi

    if [ "$metric" == "f1" ]; then
        echo "========================================="
        echo "starting to compare with knowledge for soft-f1, sql_dialect: ${sql_dialect}"
        python3 -u ./evaluation_f1.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path}  \
        --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus}  --output_log_path ${output_log_path} \
        --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}   --sql_dialect ${sql_dialect}
    fi
done