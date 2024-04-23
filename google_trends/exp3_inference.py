import argparse
import json
import numpy as np
import torch
import string
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    model_name,
    output_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("potsawee/cultural_awareness_mcq")['train']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # building vocab mapping
    # e.g., for mistral "▁C" -> 334, "C" -> 28743
    # tokenizer("C", add_special_tokens=False).input_ids -> [334]
    # here we want 28743
    vocab2id = {k: v for k, v in tokenizer.vocab.items()} # vocab2id["▁C"] -> 334, vocab2id["C"] -> 28743
    label_ids = [vocab2id[c] for c in num2letter]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    outputs = {}

    for i in tqdm(range(len(dataset))):
        question = dataset[i]['question']
        options = dataset[i]['options']
        prompt = template2prompt(question, options)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask) 
        vocab_logits = output.logits[0, -1, :]
        # debug -- vocab_logits.topk(20)
        
        num_options = len(options)
        label_ids_this_q = label_ids[:num_options]
        class_logits = vocab_logits[label_ids_this_q]
        class_probs = torch.softmax(class_logits, dim=-1)
        class_probs = class_probs.tolist()
        outputs[i] = class_probs

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
    with torch.no_grad():
        main(**kwargs)

