"""
Normalising search terms
i.e., translate from Thai to English
"""

import os
import openai
from openai import OpenAI

import argparse
import json
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # organization=os.getenv("OPENAI_ORGANIZATION"),
)


def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125') # gpt-3.5-turbo-0125
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    model = kwargs['model']
    input_path = kwargs['input_path']
    output_path = kwargs['output_path']

    print("model:", model)
    print("input_path:", input_path)
    print("output_path:", output_path)

    # LLM to be evaluated
    with open(input_path, "r") as f:
        items = json.load(f)

    system_prompt = "Your task is to process the following text into standard English. If the text is in a different language other than English, translate it into English.\n\n"
    system_prompt += "Text: youtube\n"
    system_prompt += "Response: Youtube\n\n"
    system_prompt += "Text: กรุงเทพ\n"
    system_prompt += "Response: Bangkok\n\n"
    system_prompt += "Text: uomini e donne\n"
    system_prompt += "Response: Uomini e Donne\n"
    

    for i, item in tqdm(enumerate(items)):
        text = item['object']
        prompt = f"Text: {text}\nResponse:"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        gen_text = response.choices[0].message.content
        items[i]['gpt'] = gen_text
        if i % 500 == 0:
            cache_path = output_path + f"_{i}.json"
            with open(cache_path, "w") as f:
                json.dump(items, f)
            print("write:", cache_path)
            
    final_path = output_path + "_final.json"
    with open(final_path, "w") as f:
        json.dump(items, f)
    print("write:", final_path)
            