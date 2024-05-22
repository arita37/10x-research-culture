import argparse
import os
import json
import numpy as np
import string
from tqdm import tqdm

from datasets import load_dataset
import openai
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION"),
    # api_key=os.environ.get("OPENTYPHOON_API_KEY"),
    # base_url="https://api.opentyphoon.ai/v1",
)

PROMPT_TEMPLATE = "Your task is to answer the following question by selecting one of the options 1-4. Give the answer by its letter without any explanation.\n\คำถาม: \"{word}\" หมายความว่าอะไรในภาษาไทยกลาง\n\n"
# num2letter = [c for c in string.ascii_uppercase]
num2letter = ["1", "2", "3", "4"] # jsut 4 options

def template2prompt(word, options):
    prompt = PROMPT_TEMPLATE.format(word=word)
    for i in range(len(options)):
        prompt += f"({num2letter[i]}) {options[i]}\n"
    prompt += "\nคำตอบ: ("
    return prompt


def main(
    model_name, # 'gpt-3.5-turbo-0125'
    output_path,
):
    with open("/data/workspace/exp-punpun/culture/thai_dialect_mcq/combined_560_with_distractors.json") as f:
        dataset = json.load(f)

    outputs = {}

    for i in tqdm(range(len(dataset))):
        word = dataset[i]['dialect']
        options = dataset[i]['options']
        prompt = template2prompt(word, options)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            # min_tokens=2,
            max_tokens=5,
        )
        gen_text = response.choices[0].message.content
        if len(gen_text) == 1:
            gen_text += ")))" # quick hack
        if gen_text[0] in num2letter:
            outputs[i] = gen_text[0]
        elif gen_text[1] in num2letter:
            outputs[i] = gen_text[1]
        else:
            outputs[i] = 0
            # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
    with open(output_path, "w") as f:
        json.dump(outputs, f)
    print("write:", output_path)


def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    main(**kwargs)

