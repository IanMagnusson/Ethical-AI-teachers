import json
import os
import sys
from data.mbpp import get_mbpp

def augment_eval_results(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    mbpp_data = get_mbpp()
    for q_id in data['eval']:
        task_id = q_id.split("/")[1]
        q = mbpp_data[task_id]
        assert len(data['eval'][q_id]) == 1, f"wrong format for {q_id}: {data['eval'][q_id]}"
        data['eval'][q_id][0]['prompt'] = q['prompt']
        data['eval'][q_id][0]['reference_code'] = q['code']
        data['eval'][q_id][0]['test_imports'] = q['test_imports']

    output_file = os.path.splitext(input_file)[0] + '_augmented.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Augmented results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_mpbb_to_eval_outputs.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    augment_eval_results(input_file)
