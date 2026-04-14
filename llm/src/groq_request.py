#!/usr/bin/env python3
import argparse
import json
import os
from groq import Groq
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from prompt import generate_combined_prompts_one


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def connect_groq(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the Groq API and get the response.
    """
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        try:
            messages = [
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            result = response.choices[0].message.content
            break
        except Exception as e:
            error_str = str(e)
            result = "error:{}".format(error_str)
            print(f"API Error (Attempt {i+1}/{MAX_API_RETRY}):", result)
            
            # If rate limited (e.g. 429), backoff
            if "429" in error_str or "rate limit" in error_str.lower():
                time.sleep(15)
            else:
                time.sleep(4)
    return result


def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = os.path.join(db_root_path, data["db_id"], data["db_id"] + ".sqlite")
        db_path_list.append(cur_db_path)
        knowledge_list.append(data.get("evidence", ""))

    return question_list, db_path_list, knowledge_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for i, (sql, _) in enumerate(sql_lst):
        result[i] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


def init_client(api_key):
    """
    Initialize the Groq client for a worker.
    """
    return Groq(api_key=api_key)


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else str(response)
    db_id = os.path.basename(db_path).replace(".sqlite", "")
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(task_data):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the Groq response.
    """
    prompt, engine, client, db_path, question, i = task_data
    # Groq does not strictly require stop sequence arrays in all models, but we can pass it
    response = connect_groq(engine, prompt, 1024, 0.0, ["--", "\n\n", ";", "#"], client)
    sql = post_process_response(response, db_path)
    print(f"Processed {i}th question: {question}")
    return sql, i


def collect_response_from_groq(
    db_path_list,
    question_list,
    api_key,
    engine,
    sql_dialect,
    num_threads=3,
    knowledge_list=None,
):
    """
    Collect responses from Groq using multiple threads.
    """
    client = init_client(api_key)

    tasks = [
        (
            generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                sql_dialect=sql_dialect,
                knowledge=knowledge_list[i] if knowledge_list else "",
            ),
            engine,
            client,
            db_path_list[i],
            question_list[i],
            i,
        )
        for i in range(len(question_list))
    ]
    responses = []
    
    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(worker_function, task): task for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task), total=len(tasks)
        ):
            responses.append(future.result())
            
    return responses


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, required=True)
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, required=True)
    args_parser.add_argument("--api_key", type=str, required=True)
    args_parser.add_argument("--engine", type=str, required=True, default="meta-llama/llama-4-scout-17b-16e-instruct")
    args_parser.add_argument("--data_output_path", type=str, required=True)
    args_parser.add_argument("--chain_of_thought", type=str, default="True")
    args_parser.add_argument("--num_processes", type=int, default=3)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--num_samples", type=int, default=None)
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path, "r"))
    if args.num_samples is not None:
        eval_data = eval_data[:args.num_samples]

    question_list, db_path_list, knowledge_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list)

    responses = collect_response_from_groq(
        db_path_list,
        question_list,
        args.api_key,
        args.engine,
        args.sql_dialect,
        args.num_processes,
        knowledge_list if args.use_knowledge == "True" else None,
    )

    if args.chain_of_thought == "True":
        output_name = os.path.join(
            args.data_output_path,
            f"predict_{args.mode}_{args.engine.replace('/', '_')}_cot_{args.sql_dialect}.json"
        )
    else:
        output_name = os.path.join(
            args.data_output_path,
            f"predict_{args.mode}_{args.engine.replace('/', '_')}_{args.sql_dialect}.json"
        )
        
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )
