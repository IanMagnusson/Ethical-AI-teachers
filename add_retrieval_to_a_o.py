import argparse
import pandas as pd
import json

def main(input_file, retrieval_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Read the JSONL file
    with open(retrieval_file) as f:
        data = [json.loads(line) for line in f]

    # Process the retrieval data
    id_to_retrieved = {d["qid"]: d["search_results"] for d in data}
    for qid, retrieved in id_to_retrieved.items():
        assert len(retrieved) == 2 and retrieved[1] is None
        id_to_retrieved[qid] = id_to_retrieved[qid][0]
        if len(id_to_retrieved[qid]) != 10:
            assert len(id_to_retrieved[qid]) > 0

    # Map the retrieved data to the dataframe
    df['retrieved'] = df['task_id'].map(id_to_retrieved)

    # Write the output CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input and retrieval files.')
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--retrieval_file', type=str, help='Path to the retrieval JSONL file')
    parser.add_argument('--output_file', type=str, help='Path to the output CSV file')

    args = parser.parse_args()

    main(args.input_file, args.retrieval_file, args.output_file)
