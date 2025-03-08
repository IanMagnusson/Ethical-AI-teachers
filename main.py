import argparse

from codegen import run_codegen
from data.mbpp import get_mbpp_plus
from evaluate import evaluate

def main(args):
    model_name = args.model
    num_gpus = args.num_gpus
    greedy = args.greedy
    mbpp_dataset = get_mbpp_plus(mini=False, noextreme=False, version="default")
    debug = args.debug
    root = args.root
    
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
        
        # take only the first 10 tasks
        mbpp_dataset = {k: mbpp_dataset[k] for k in mbpp_keys[:10]}
    
    # TODO: here, we might want to do something to augment the mbpp dataset in accordance with our approach.
        
    samples_path = run_codegen(model_name, tp=num_gpus, dataset="mbpp", dataset_files=mbpp_dataset, backend="vllm", greedy=greedy, debug=debug, root=root)
    
    breakpoint()
        
    evaluate(
        dataset="mbpp",
        samples=samples_path,
        debug=debug,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=1, help="Num of GPUs to use, passed to vllm as tensor_parallel_size")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--root", type=str, default="mbpp_results", help="Root directory for saving results")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    main(args)
