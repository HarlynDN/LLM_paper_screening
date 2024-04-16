# LLM Paper Screening

Given a list of papers, this script uses LLMs to automatically screen papers that are relevant to the given field(s). 

The prompt and LLM output are in Chinese.

## Workflow
For each paper in the given list:
1. Search the paper on arxiv
2. If found, obtain the abstract of the paper, and prompt LLM to do chain-of-thoughts reasoning:
    - briefly summarize the paper
    - reason about the paper's relevance to the given field(s)
    - return a subset of the given fields that are relevant to the paper

Several demonstrations will be provided in context to elicit the reasoning.


## Requirements
```txt
transformers>=4.37.0
arxiv
```

## Example Usage
```python
python run.py \
    --paper naacl24_papers.json \
    --field field.txt \
    --inst instruction.txt \
    --demo demos.json \
    --model Qwen/Qwen1.5-32B-Chat \
    --device cuda
```

`--paper` should be a json file containing a list of dictionaries. Each dictionary should have a key "title" and an optional key "authors". This information will be used to search the paper on arxiv.

`--field` should be a text file containing field(s) of interest. Each field should be on a separate line.

`--model` supports Qwen1.5 series models by default. Other models can be easily integrated by modifying the prompt formatting in `run.py`

Please refer to `run.py` for more details on the arguments.
