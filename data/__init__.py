import json

from datasets import load_dataset

from data.mbpp import get_mbpp_plus, get_mbpp_plus_hash
from data.utils import load_solutions, write_directory, write_jsonl


def get_evalperf_data():
    dataset = load_dataset("evalplus/evalperf", split="test").to_list()
    for d in dataset:
        d["pe_input"] = json.loads(d["pe_input"])
    return {task["task_id"]: task for task in dataset}