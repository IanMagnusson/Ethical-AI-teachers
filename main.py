import argparse
import random

import numpy as np
import pandas as pd
import torch

from codegen import run_codegen
from data.mbpp import get_mbpp_plus
from evaluate import evaluate
import json

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    model_name = args.model
    num_gpus = args.num_gpus
    mbpp_dataset = get_mbpp_plus(mini=False, noextreme=False, version="default")
    debug = args.debug
    root = args.root
    use_mini = args.use_mini
    feedback_file = args.feedback_file
    seed = args.seed
    
    set_seed_everywhere(seed)
    
    if feedback_file:
        print(f"Loading feedback from {feedback_file}")
        feedback_df = pd.read_csv(feedback_file)
        if args.retrieval_as_feedback:
            feedback_df["retrieved"] = feedback_df["retrieved"].apply(eval).apply(lambda x: x[:3]).apply(lambda x: json.dumps(x, indent=2))
            feedback_df["feedback"] = feedback_df["retrieved"]
        feedback_df_task_ids = feedback_df["task_id"].values
        
        for key in mbpp_dataset.keys():
            if key in feedback_df_task_ids:
                mbpp_dataset[key]["feedback"] = feedback_df.loc[feedback_df["task_id"] == key, "feedback"].values[0]
                mbpp_dataset[key]["init_soln"] = feedback_df.loc[feedback_df["task_id"] == key, "solution"].values[0] if not args.feedback_only else None
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
                                    
    samples_path = run_codegen(model_name, tp=num_gpus, dataset="mbpp", dataset_files=mbpp_dataset, backend="vllm", greedy=True, debug=debug, root=root, use_mini=use_mini, seed=seed)
            
    evaluate(
        dataset="mbpp",
        samples=samples_path,
        debug=debug,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    
    # Model
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=1, help="Num of GPUs to use, passed to vllm as tensor_parallel_size")
    
    # Save folder
    parser.add_argument("--root", type=str, default="mbpp_results", help="Root directory for saving results")
    
    # Feedback
    parser.add_argument("--feedback_file", type=str, default=None, help="File to load feedback from")
    
    # For debugging
    parser.add_argument("--use_mini", action="store_true", help="Use mini dataset")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--feedback_only", action="store_true", help="Condition on just the feedback without the questions or failed guess.")
    parser.add_argument("--retrieval_as_feedback", action="store_true", help="Use retrieval as feedback")
    args = parser.parse_args()
    main(args)
