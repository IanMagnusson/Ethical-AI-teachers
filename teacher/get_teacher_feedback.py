import os
import argparse
import pandas as pd
import openai
from tqdm import tqdm
import json

def check_openai_quota(api_key):
    client = openai.Client(api_key=api_key)
    try:
        response = client.models.list()
        print("API Key is valid, and quota exists.")
        return True
    except openai.AuthenticationError as e:
        print(f"Invalid API Key: {e}")
    except openai.RateLimitError:
        print("API Key is valid, but quota is exhausted.")
    except openai.APIError as e:
        print(f"An OpenAI API error occurred: {e}")
    return False

def get_teacher_feedback(client, teacher_system_prompt, teacher_prompt, model, max_tokens=1000):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": teacher_system_prompt},
            {"role": "user", "content": teacher_prompt},
        ],
        max_tokens=max_tokens,
        temperature=1,
        top_p=1
    )
    return response

def build_prompt(student_prompt, student_answer, base_fail_tests, plus_fail_tests, reference_code):
    teacher_prompt = "Please help me solve this problem."
    teacher_prompt += f"\n\nOriginal question:\n{student_prompt}\n\nStudent answer:\n{student_answer}"
    teacher_prompt += f"\n\nFailed inputs:\n{base_fail_tests + plus_fail_tests}"
    teacher_prompt += f"\n\nWhat feedback can you give me to help me solve this problem?\n\n"
    return teacher_prompt

def get_feedback_for_all_errors(df, client, model, teacher_system_prompt):
    assert all((df['base_status'] == 'fail') | (df['plus_status'] == 'fail')), "All rows must have at least one failing test"
    df = df.copy()
    df['feedback'] = None
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Getting feedback"):
        prompt = build_prompt(row['question_prompt'], row['solution'], row['base_fail_tests'], row['plus_fail_tests'], row['gt_solution'])
        response = get_teacher_feedback(client, teacher_system_prompt, prompt, model)
        assert len(response.choices) == 1, "Expected exactly one response"
        feedback = response.choices[0].message.content
        df.at[i, 'feedback'] = feedback
    return df

def main(args):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

    client = openai.Client(api_key=OPENAI_API_KEY)
    if not check_openai_quota(OPENAI_API_KEY):
        return

    df = pd.read_csv(args.input_filepath)
    teacher_system_prompt = "You are a Python tutor. A student has written the following code, but it is not working as expected. Provide feedback to help the student fix the code."
    df_with_feedback = get_feedback_for_all_errors(df, client, args.model, teacher_system_prompt)

    output_filepath = args.input_filepath.replace('.csv', f'_with_feedback_{args.model}.csv')
    df_with_feedback.to_csv(output_filepath, index=False)
    print(f"Feedback saved to {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file and get feedback for errors using OpenAI API.")
    parser.add_argument("--input_filepath", type=str, help="Path to the input CSV file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    args = parser.parse_args()
    main(args)
