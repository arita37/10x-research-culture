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
    # organization=os.getenv("OPENAI_ORGANIZATION"),
)

PROMPT_TEMPLATE = "Your task is to answer the following question by selecting one of the options. Give the answer by its letter without any explanation.\n\nQuestion: {question}\n\n"
# num2letter = [c for c in string.ascii_uppercase]
num2letter = ["A", "B", "C", "D"] # jsut 4 options

def template2prompt(question, options):
    prompt = PROMPT_TEMPLATE.format(question=question)
    for i in range(len(options)):
        prompt += f"({num2letter[i]}) {options[i]}\n"
    prompt += "\nAnswer: ("
    return prompt


def main(
    model_name, # 'gpt-3.5-turbo-0125'
    output_path,
):
    dataset = load_dataset("potsawee/cultural_awareness_mcq")['train']


    outputs = {}

    for i in tqdm(range(len(dataset))):
        question = dataset[i]['question']
        options = dataset[i]['options']
        prompt = template2prompt(question, options)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        gen_text = response.choices[0].message.content
        if gen_text[0] in num2letter:
            outputs[i] = gen_text[0]
        elif gen_text[1] in num2letter:
            outputs[i] = gen_text[1]
        else:
            import ipdb; ipdb.set_trace()
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

