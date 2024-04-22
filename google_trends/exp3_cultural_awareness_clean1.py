"""
Experiment3: Cultural Awareness

(Answer, Distractors, Entity, Category) -> Question

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
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125') # gpt-3.5-turbo-0125, gpt-4-turbo-2024-04-09
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    return parser


def main(args):
    model = args['model']
    category = args['category']
    output_path = args['output_path']

    print("model:", model)
    print("category:", category)
    print("output_path:", output_path)

    if os.path.isfile(output_path):
        print("file exists:", output_path)
        return
    
    input_path = f"data/exp3_cultural_awareness_v2/{category}.json"
    with open(input_path, "r") as f:
        items = json.load(f)

    system_prompt = "You are a helpful assistant. Your task is to decide whether the entity is related to the topic or not."
    
    count_yes = 0
    count_no = 0
    good_items = []
    for i, item in enumerate(items):
        category = item['category']
        entity = item['entity']
        prompt = 'Is the entity related to the topic or not. You must answer "Yes" if the entity is related. '
        prompt += 'Otherwise you must answer "No". You must not provide any explaination, just say Yes or No.\n\n'
        prompt += f"Topic: {category}\n" 
        prompt += f"Entity: {entity}\n\n" 
        prompt += "Answer (Yes or No):"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=10,
        )
        gen_text = response.choices[0].message.content
        if gen_text == 'Yes':
            good_items.append(item)
            count_yes += 1
        elif gen_text == 'No':
            count_no += 1
        else:
            import ipdb; ipdb.set_trace()
        percentage = (count_yes)/(count_yes+count_no)
        print("[{}/{}]: good = {:.2f}% --- {} => {} [{}]".format(i+1, len(items), percentage*100, category, entity, gen_text))
   
    with open(output_path, "w") as f:
        json.dump(good_items, f)
    print("write:", output_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    main(kwargs)