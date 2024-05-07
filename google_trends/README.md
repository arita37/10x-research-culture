# Google Trends

## Idea
- using most searched terms from Google Trends as proxy about people's interests - cr. Yoyo

## Scripts
- `exp3` = cuurent MCQ experiment using https://huggingface.co/datasets/potsawee/cultural_awareness_mcq
  - `exp3_cultural_awareness_clean1.py` = clean data using GPT
  - `exp3_cultural_awareness_gen_q.py` = generate questions using GPT
  - `exp3_inference.py` = running inference using open-source model to get logits (and probs)
  - `exp3_inference_gpt.py` = running inference using OpenAI's API to get logits (and probs)
 
## Data
- `data/internal_processed_gpt_clean.json` = GPT translated data into English
- `data/exp3_cultural_awareness_v2/clean1` = after applying `exp3_cultural_awareness_clean1.py`
- `data/exp3_cultural_awareness_v2/MCQ` = after mapping to multiple-choice questions
