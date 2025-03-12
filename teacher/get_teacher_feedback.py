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

def build_prompt(student_prompt, student_answer, base_fail_tests, plus_fail_tests, reference_code, retrieved=None):
    teacher_prompt = "Please help me solve this problem."
    teacher_prompt += f"\n\nOriginal question:\n{student_prompt}\n\nStudent answer:\n{student_answer}"
    teacher_prompt += f"\n\nFailed inputs:\n{base_fail_tests + plus_fail_tests}"
    if retrieved:
        retrieved = eval(retrieved)
        retrieved = retrieved[:3]
        teacher_prompt += f"\n\nPotentially relevant retrieved examples:\n{json.dumps(retrieved, indent=2)}"
    teacher_prompt += f"\n\nWhat feedback can you give me to help me solve this problem?\n\n"
    return teacher_prompt

def get_feedback_for_all_errors(df, client, model, teacher_system_prompt):
    assert all((df['base_status'] == 'fail') | (df['plus_status'] == 'fail')), "All rows must have at least one failing test"
    df = df.copy()
    df['feedback'] = None
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Getting feedback"):
        retrieved = (row['retrieved'] if 'retrieved' in df.columns else None)
        prompt = build_prompt(row['question_prompt'], row['solution'], row['base_fail_tests'], row['plus_fail_tests'], row['gt_solution'], retrieved=retrieved)
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
    if args.no_question_prompt:
        teacher_system_prompt += "\n\nPlease do not repeat the question prompt. Be concise and just give feedback on the student's answer."
    if args.no_cheat_prompt:
        teacher_system_prompt += "\n\nPlease do not provide the exact solution. Instead, provide feedback that will help the student learn."
    df_with_feedback = get_feedback_for_all_errors(df, client, args.model, teacher_system_prompt)

    output_filepath = args.input_filepath.replace('.csv', f'_with_feedback_{args.model}.csv')
    if args.output_dir:
        output_filepath = os.path.join(args.output_dir, os.path.basename(output_filepath))
    df_with_feedback.to_csv(output_filepath, index=False)
    print(f"Feedback saved to {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file and get feedback for errors using OpenAI API.")
    parser.add_argument("--input_filepath", type=str, help="Path to the input CSV file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--no_cheat_prompt", action="store_true", help="Do include the prompt not to cheat")
    parser.add_argument("--no_question_prompt", action="store_true", help="ask the model not to repeat the question prompt")
    parser.add_argument("--output_dir", type=str, help="Path to save the output CSV")
    args = parser.parse_args()
    main(args)
