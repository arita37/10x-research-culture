"""
(Dialect, Central, Type) -> Distractors
"""

import os
import openai
from openai import OpenAI

import argparse
import json
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", bool, lambda v: v.lower() == "true")
    parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13') # gpt-4o-2024-05-13
    parser.add_argument('--output_dir', type=str, required=True)
    return parser


def main(args):
    model = args['model']
    output_dir = args['output_dir']
    verbose = True

    print("model:", model)
    print("output_dir:", output_dir)
    
    input_path = "./combined_560.json"
    with open(input_path, "r") as f:
        items = json.load(f)

    system_prompt = "You are a helpful assistant that help generate distractors for multiple-choice questions.\n"
    system_prompt += "Distractors are plausible choices for the question and answer, but they are not correct.\n"
    system_prompt += "You will be given a Thai word in dialect (e.g., Northern, Southern, or Isarn) and its meaning, your task is to generate 3 distractors, which are plausible a meaning of the word. Distractors should be related to the correct meaning, but the distractors are incorrect.\n" 
    system_prompt += "You must return 3 distractors in a list, and the distractors must be in the Thai language (ภาษาไทย).\n\n"
    system_prompt += "The following are some examples of distractors.\n\n"
    system_prompt += "Word (southern dialect): ต่อเช้า\n"
    system_prompt += "Meaning: พรุ่งนี้\n"
    system_prompt += "Distractors: [\n"
    system_prompt += "'เมื่อวาน',\n"
    system_prompt += "'เย็นนี้',\n"
    system_prompt += "'ตอนเช้า',\n"
    system_prompt += "]\n\n"
    system_prompt += "Word (northern dialect): สายฮั้ง\n"
    system_prompt += "Meaning: เข็มขัด\n"
    system_prompt += "Distractors: [\n"
    system_prompt += "'สายวัด',\n"
    system_prompt += "'กระโปรง',\n"
    system_prompt += "'ผ้าห่ม',\n"
    system_prompt += "]\n\n"
    system_prompt += "Word (isarn dialect): จั๊กเด่ะ\n"
    system_prompt += "Meaning: ไม่รู้\n"
    system_prompt += "Distractors: [\n"
    system_prompt += "'ชัดเจน',\n"
    system_prompt += "'ง่ายมาก',\n"
    system_prompt += "'ใจดี',\n"
    system_prompt += "]\n\n"
    
    mapping = {
        'northern': 'northern',
        'southern': 'southern',
        'nangrong': 'isarn',
    }
    
    for i, item in enumerate(items):
        output_path = f"{output_dir}/{i}.json"
        if os.path.isfile(output_path):
            print("file exist:", output_path)
            continue        

        word = item['dialect']
        meaning = item['central']
        dialect_type = item['type']
        prompt = "Word ({} dialect): {}\n".format(mapping[dialect_type], word)
        prompt += "Meaning: {}\n".format(meaning)
        prompt += "Distractors:"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        gen_text = response.choices[0].message.content
        try:
            distractors = eval(gen_text)
        except:
            import ipdb; ipdb.set_trace()
        this_item = {
            "source": item['source'],
            "type": dialect_type,
            "dialect": word,
            "central": meaning,
            "distractor": distractors,
        }
        if verbose:
            print(word, " ------> ", meaning, distractors)
            print("---------------------------------------------------------------")
        with open(output_path, "w") as f:
            json.dump(this_item, f, indent=4, ensure_ascii=False)
        print("write:", output_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    kwargs = vars(parser.parse_args())
    main(kwargs)