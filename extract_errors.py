import os
import json
import pandas as pd
from data import get_mbpp_plus
from argparse import ArgumentParser
from data.utils import CACHE_DIR
from config import *

import pandas as pd

from eval.utils import (
    swallow_io,
    time_limit,
)

def execute_python_code_with_inputs(code, entry_point, inp, time):
    exec_globals = {}
    with swallow_io():
        exec(code, exec_globals)
        fn = exec_globals[entry_point]

        try:
            with time_limit(time):
                with swallow_io():
                    out = fn(*inp)
        except Exception as e:
            out = str(e)
    return out

def convert_tuples_to_lists(data):
    if isinstance(data, tuple):  
        return [convert_tuples_to_lists(item) for item in data] 
    elif isinstance(data, list):  # If it's a list, process it recursively
        return [convert_tuples_to_lists(item) for item in data]
    elif isinstance(data, dict):  # If it's a dictionary, process values
        return {key: convert_tuples_to_lists(value) for key, value in data.items()}
    return data  


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--results_folder", type=str, default="mbpp_results/mbpp")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--output_format", type=str, default="csv")
    return parser.parse_args()


def main():
    args = parse_args()

    model_str = args.model.replace("/", "--")
    model_results_file = os.path.join(args.results_folder, f"{model_str}_vllm_temp_0.0.eval_results.json")
   
    model_results = json.load(open(model_results_file))
    hashcode = model_results['hash']
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")

    # if not os.path.exists(cache_file):
    #     raise ValueError(f"Cache file {cache_file} not found")
    # else:
    expected_outputs = pd.read_pickle(cache_file)
    for key, value in expected_outputs.items():
        value['task_id'] = key
    exp_output_df = pd.DataFrame(list(expected_outputs.values()))

    results = [result[0] for result in model_results['eval'].values()]
    df = pd.DataFrame(results)

    # get failure cases
    failed_df = df[(df['base_status'] == "fail") | (df['plus_status'] == "fail")]
    print(f"{args.model}: failed {len(failed_df)} out of {len(df)}")

    # get original problems
    problems = get_mbpp_plus()
    problems_df = pd.DataFrame(list(problems.values()))

    # join on task id 
    full_error_df = failed_df.merge(problems_df, left_on='task_id', right_on='task_id')

    # remove/rename columns for clarity
    full_error_df = full_error_df.drop(columns=['contract', 'assertion', 'atol'])
    full_error_df = full_error_df.rename(columns={'canonical_solution': 'gt_solution'})
    
    for i, row in full_error_df.iterrows():
        exp_output = exp_output_df[exp_output_df['task_id'] == row['task_id']]
        problem = problems_df[problems_df['task_id'] == row['task_id']]
        # extract the code snippet
        code_snippet = row['solution']
        
        for subset in ['base', 'plus']:
            ref_times = exp_output[f'{subset}_time'].values[0]
            time_limits = [max(DEFAULT_MIN_TIME_LIMIT, DEFAULT_GT_TIME_LIMIT_FACTOR * t) for t in ref_times]

            # extract the inputs
            fail_inputs = row[f'{subset}_fail_tests']
            entry_point = row['entry_point']
            pred_outputs = []
            gt_outputs = []
            for j, input in enumerate(fail_inputs):
                all_test_inputs = list(problem[f'{subset}_input'].values[0])

                all_test_inputs = [convert_tuples_to_lists(input) for input in all_test_inputs]
                if input in all_test_inputs:
                    test_id = all_test_inputs.index(input)

                    # execute the code snippet with the inputs
                    time_limit = time_limits[j]
                    result = execute_python_code_with_inputs(code_snippet, entry_point, input, time_limit)
                    pred_outputs.append(result)

                    gt_output = exp_output[f'{subset}'].values[0][test_id]
                    gt_outputs.append(gt_output)

                else:
                    print(f"Error {row['task_id']}: {input} not found in test inputs")
                    pred_outputs.append(None)
                    gt_outputs.append(None)

            assert len(pred_outputs) == len(fail_inputs), f"Length mismatch: {len(pred_outputs)} != {len(fail_inputs)}"
            assert len(pred_outputs) == len(gt_outputs), f"Length mismatch: {len(pred_outputs)} != {len(gt_outputs)}"
            full_error_df.loc[i, f'{subset}_pred_outputs'] = str(pred_outputs)
            full_error_df.loc[i, f'{subset}_gt_outputs'] =  str(gt_outputs)
    
    # save to file
    print(f"Saving {len(full_error_df)} examples")
    if args.output_format == "csv":
        full_error_df.to_csv(os.path.join(args.results_folder, f"{model_str}_errors.csv"), index=False)
    elif args.output_format == "json":
        full_error_df.to_json(os.path.join(args.results_folder, f"{model_str}_errors.json"), orient='records')
    else:
        raise ValueError(f"Invalid output format: {args.output_format}")


if __name__ == "__main__":
    main()