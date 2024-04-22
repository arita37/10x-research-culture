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
    verbose = True

    print("model:", model)
    print("category:", category)
    print("output_path:", output_path)

    if os.path.isfile(output_path):
        print("file exists:", output_path)
        return
    
    input_path = f"data/exp3_cultural_awareness_v2/clean1/{category}.json"
    with open(input_path, "r") as f:
        items = json.load(f)

    system_prompt = "Your task is to write a question for the following scenario and the provided answer. The question must be at least one sentence and shorter than five sentences. Do not provide any explanation."
    
    count_na = 0

    for i, item in enumerate(items):
        entity = item['entity']
        answer = item['answer']
        prompt = "You are provided with a topic and an entity. You are also given the target answer and 3 distractors. Your task is to write a question related to the topic and the entity.\n\n"
        prompt += f"Topic: {item['category']}\n" 
        prompt += f"Entity: {entity}\n" 
        prompt += f"Answer: {answer}\n" 
        prompt += f"Distractor: {item['distractors'][0]}, {item['distractors'][1]}, {item['distractors'][2]}\n\n" 
        prompt += "The answer of this question must be the target answer and not a distractor."
        prompt += f' Note that "{entity}" is a popular search term in "{answer}".'
        prompt += " Your question must not include the target answer or any of the distractors, and the question should be relevant to the topic and the entity."
        prompt += "Question:"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=120,
        )
        gen_text = response.choices[0].message.content
        items[i]['gen_question'] = gen_text
        if gen_text == 'N/A':
            count_na += 1
        percentage = (i+1-count_na)/(i+1)
        print("[{}/{}]: good = {:.2f}%".format(i+1, len(items), percentage*100))
        if verbose:
            print(gen_text)
            print(entity, answer, item['distractors'])
            print("---------------------------------------------------------------")
    with open(output_path, "w") as f:
        json.dump(items, f)
    print("write:", output_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    main(kwargs)