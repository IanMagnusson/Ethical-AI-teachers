import argparse
import json
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mbpp import get_mbpp_plus
from retriever.youcom_api import search_youcom


def load_cached_results(path):
    if not os.path.exists(path):
        return []
    
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            ex = json.loads(line)
            data.append(ex)
    return data

def main(args):
    model_name = args.model
    num_gpus = args.num_gpus
    mbpp_dataset = get_mbpp_plus(mini=False, noextreme=False, version="default")
    debug = args.debug
    root = args.root
    use_mini = args.use_mini
    feedback_file = args.feedback_file
    
    if feedback_file:
        print(f"Loading feedback from {feedback_file}")
        feedback_df = pd.read_csv(feedback_file)
        feedback_df_task_ids = feedback_df["task_id"].values
        
        for key in mbpp_dataset.keys():
            if key in feedback_df_task_ids:
                mbpp_dataset[key]["feedback"] = feedback_df.loc[feedback_df["task_id"] == key, "feedback"].values[0]
                mbpp_dataset[key]["init_soln"] = feedback_df.loc[feedback_df["task_id"] == key, "solution"].values[0]
            else:
                mbpp_dataset[key]["feedback"] = None
                mbpp_dataset[key]["init_soln"] = None
    else:
        for key in mbpp_dataset.keys():
            mbpp_dataset[key]["feedback"] = None
            mbpp_dataset[key]["init_soln"] = None
    
    if debug:
        print(f"Example of MBPP Dataset Item:")
        print("=" * 100)
        print(f"type(mbpp_dataset): {type(mbpp_dataset)}")
        mbpp_keys = list(mbpp_dataset.keys())
        zero_key = mbpp_keys[0]
        zero_item = mbpp_dataset[zero_key]
        zero_item_keys = zero_item.keys()
        print(f"type(mbpp_dataset[0]): {type(zero_item)}")
        print(f"mbpp_dataset[0].keys(): {zero_item_keys}")
        for key in zero_item_keys:
            print(f"{key}: {zero_item[key]}")
        print("=" * 100)
        
    if use_mini:
        # take only the first 10 tasks
        mbpp_dataset = {k: mbpp_dataset[k] for k in mbpp_keys[:10]}
                                    
    
    # retrieve based on original prompt
    os.makedirs(os.path.dirname(args.search_result_save_path), exist_ok=True)
    cached_results = load_cached_results(args.search_result_save_path)
    cached_qids = [ex['qid'] for ex in cached_results]
    
    for qid in mbpp_dataset.keys():
        if qid in cached_qids:
            continue
        original_prompt = mbpp_dataset[qid]["prompt"]
        search_results = search_youcom(original_prompt)
        # mbpp_dataset["youcom_searched_results"] = search_results
        with open(args.search_result_save_path, "a+") as fout:
            result = {
                'qid': qid,
                'search_results': search_results,
            }
            fout.write(json.dumps(result)+'\n')
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=1, help="Num of GPUs to use, passed to vllm as tensor_parallel_size")
    
    # Save folder
    parser.add_argument("--root", type=str, default="mbpp_results", help="Root directory for saving results")
    parser.add_argument("--search_result_save_path", type=str, default="mbpp_results/youcom_search.jsonl", help="Root directory for saving results")
    
    # Feedback
    parser.add_argument("--feedback_file", type=str, default=None, help="File to load feedback from")
    
    # For debugging
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    main(args)
