import json
import os
import logging
from typing import Dict, List
import argparse
import arxiv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument('--paper', type=str,
                    help='path to the paper list, should be a json file containing a list dictionaries. Each dictionary should have a key "title" and an optional key "authors"')

parser.add_argument('--inst', type=str, default='instruction.txt', help='path to the instruction file')

parser.add_argument('--field', type=str, default='field.txt', help='path to a .txt file where each line is a target field name')

parser.add_argument('--demo', type=str, default='demos.json', help='path to the demonstration file')

parser.add_argument('--model', type=str, default='Qwen/Qwen1.5-32B-Chat', help='model name or path')

parser.add_argument('--device', type=str, default='cuda', 
                    help='if set to "cuda", will use single GPU; if set to "auto", will shard the model across all available GPUs; if set to "cpu", will use CPU')


NOT_FOUND = {'error': 'not found'}

def match(res: arxiv.Result, title: str, authors: list = []):
    success = len(set(title.split()).intersection(res.title.split())) / len(title.split()) > 0.7
    if authors:
        res_authors = set([a.name for a in res.authors])
        success = success and len(set(authors).intersection(res_authors)) / len(set(authors)) > 0.7
    return success

def search(title: str, authors: list = []):
    client = arxiv.Client()
    query = title + ' ' + ' '.join(authors)
    search = arxiv.Search(
        query=query, max_results=10, sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)
    try:
        for res in results:
            if match(res, title, authors):
                return {'title': res.title, 'authors': [a.name for a in res.authors], 'url': res.entry_id, 'abstract': res.summary}
    except arxiv.UnexpectedEmptyPageError:
        logger.error(f"UnexpectedEmptyPageError: {title}")

    return NOT_FOUND


def make_conversation(instruction: str, demos: List[Dict]=[], sample: Dict=None):
    def _format(item):
        return json.dumps(item, indent=4, ensure_ascii=False) if isinstance(item, dict) else item if isinstance(item, str) else str(item)
    conversations = [{'role': 'system', 'content': instruction}]
    for demo in demos:
        conversations.append({'role': 'user', 'content': _format(demo['input'])})
        conversations.append({'role': 'assistant', 'content': _format(demo['output'])})
    if sample:
        conversations.append({'role': 'user', 'content': _format(sample)})
    return conversations

def update_log(log, path='output/output_log.json'):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(log, f, indent=4, ensure_ascii=False)


def run():
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    # Load data
    papers = json.load(open(args.paper))
    target_fields = [f.strip() for f in open(args.field).readlines() if f.strip()]
    instruction = open(args.inst).read()
    instruction = instruction.replace('{FIELDS}', '\n'.join(target_fields))
    demos = json.load(open(args.demo))

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto', device_map=args.device)

    # print prompt
    logger.info(f"******** DEMONSTRATIONS ********")
    for msg in  make_conversation(instruction, demos):
        logger.info(f"===== {msg['role']} =====")
        logger.info(msg['content'])
    logger.info(f"********************************")

    output_log = {
        'args': args.__dict__,
        'paper_not_found_cnt': 0,
        'format_err_cnt': 0,
        'total_cnt': 0,
        'outputs':[]
    }
    for i, paper in tqdm(enumerate(papers), total=len(papers)):
        if os.path.exists('output_log.json'):
            output_log = json.load(open('output_log.json'))
        if i < output_log['total_cnt']:
            continue
        output_log['total_cnt'] += 1

        logger.info(f"Paper not found so far: {output_log['paper_not_found_cnt']}")
        logger.info(f"Format error so far: {output_log['format_err_cnt']}\n")
    
        logger.info(f"===== Paper: {paper['title']} =====")
        # search paper on arxiv
        if 'authors' in paper.keys():
            authors = paper['authors'] if isinstance(paper['authors'], list) else [a.strip() for a in paper['authors'].split(',') if a.strip()]
        else:
            authors = []
        res = search(paper['title'], authors)
        if res == NOT_FOUND:
            logger.info(f"Paper not found!")
            output_log['paper_not_found_cnt'] += 1
            update_log(output_log)
            continue

        logger.info(f"Abstract:\n{res['abstract']}\n\n")
        # prepare prompt
        converations = make_conversation(instruction, demos, res)
        prompt = tokenizer.apply_chat_template(
                converations, tokenize=False, add_generation_prompt=True
            )
        # generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=300)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Generation:\n{generation}\n\n")

        # format check
        def is_valid_field(field: str, target_fields: List[str]=target_fields):
            for fd in target_fields:
                if field in fd:
                    return True
            return False
        try: 
            output = json.loads(generation)
            assert 'field' in output.keys() and 'summary' in output.keys()
        except:
            logger.error(f"Format error!")
            output_log['format_err_cnt'] += 1
            update_log(output_log)
            continue

        output_log['outputs'].append({
            'title': res['title'],
            'authors': res['authors'],
            'url': res['url'],
            'abstract': res['abstract'],
            'summary': output['summary'],
            'reasoning': output['reasoning'],
            'field': output['field'] 
        })

        update_log(output_log)

    logger.info(f"****** Finished ******")

    logger.info(f"Paper not found: {output_log['paper_not_found_cnt']}/{len(papers)}")
    logger.info(f"Format error: {output_log['format_err_cnt']}/{len(papers)}")


    final_outputs = []
    for item in output_log['outputs']:
        valid_target_fields = [d for d in item['field'] if is_valid_field(d)]
        if valid_target_fields:
            item['field'] = valid_target_fields
            del item['abstract']
            del item['reasoning']
            final_outputs.append(item)
    logger.info(f"Final outputs: {len(final_outputs)}")

    with open('output/outputs.json', 'w') as f:
        json.dump(final_outputs, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    run()