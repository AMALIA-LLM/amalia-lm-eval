#!/usr/bin/env python3

import math, sys, json, transformers, pandas as pd, numpy as np
from os import path, makedirs

from transformers import PreTrainedTokenizer
from typing import cast
from glob import glob
from itertools import batched

class Classifier:
    model_name : str
    pipe  : transformers.Pipeline
    max_length : int

    def __init__(self, model_name : str, supports_batching=False):
        self.model_name = model_name
        self.pipe = transformers.pipeline('text-classification', model=model_name, top_k=None)
        # self.pipe = transformers.pipeline('text-classification', model=model_name, return_all_scores=True)
        self.max_length = self.pipe.model.config.max_position_embeddings or 4096
        self.supports_batching = supports_batching

def classify_responses(cls : Classifier, responses) -> list[dict]:
    def mean_entries(values : list[dict]): 
        keys = values[0].keys()

        return {
            key: np.mean([v[key] for v in values]) for key in keys
        }

    def split_response(response : str, parts : int) -> list[str]:
        words       = response.split(' ')
        split_size  = math.ceil(len(words) / parts)
        return [' '.join(p) for p in batched(words, split_size)]

    tokenizer = cast(PreTrainedTokenizer, cls.pipe.tokenizer)

    entries   = []
    indexes   = []

    for res in responses:
        indexes.append(len(entries))
        ids = tokenizer.encode(res)

        max_tokens_nr = cls.max_length
        if len(ids) <= max_tokens_nr:
            entries.append(res)
            continue

        splits_nr   = math.ceil(len(ids) / max_tokens_nr)
        new_entries = split_response(res, splits_nr)

        while any(map(lambda e : len(tokenizer.encode(e)) > max_tokens_nr, new_entries)):
            splits_nr += 1
            new_entries = split_response(res, splits_nr)

        entries.extend(new_entries)

    indexes.append(len(entries))

    results = [
        { entry['label'] : entry['score'] for entry in score } for score in cast(list[dict], cls.pipe(entries))
    ]

    scores = [
        mean_entries(results[indexes[i]: indexes[i+1]]) for i in range(len(indexes) - 1)
    ]

    assert len(scores) == len(responses)
    return scores

def calculate_scores(classifiers : list[Classifier], responses : list[str]) -> list[list[dict]]: 
    def normalize_score(score : dict) -> float:
        base_score = score.get('PT-PT', None)
        return score['PT'] if base_score is None else base_score
        # return int(round(10 * (1 - base_score), 0))

    scores = [ 
        {
            'classifier': cls.model_name.split('/')[-1],
            'scores'    : classify_responses(cls, responses)
        } 
        for cls in classifiers
    ]

    scores  = [
        [
            { 
                entry['classifier']: val, 
                f'{entry['classifier']}_score': normalize_score(val)
            } for val in entry['scores']
        ] for entry in scores
    ]

    assert len(scores) == 2

    return scores

def main():
    if len(sys.argv) < 2:
        print(f'usage: python {sys.argv[0]} <base-dir>',file=sys.stderr)
        sys.exit(1)

    base_dir = sys.argv[1]
    output_dir = path.join(base_dir, 'class_scores')
    makedirs(output_dir, exist_ok=True)
    
    classifiers = [
        Classifier("liaad/PtVId"), 
        Classifier("bastao/PeroVaz_PT-BR_Classifier")
    ]

    print(f"> loading files from '{base_dir}'")
    for filename in glob(f'{base_dir}/*.json'):
        print(f">> loading '{filename}'")
        is_pt_pt = filename.endswith('_pt-pt.json')
        base_name = path.basename(filename).rstrip('.json') + '_scored_v1.csv'
        out_filename = path.join(output_dir, base_name)

        if path.isfile(out_filename):
            print(f">> skipping '{filename}' ...")
            continue

        with open(filename) as file:
            data = json.load(file)

        results = []
        for row in data:
            turns = row.pop('turns')
            results.extend(
               row | { 
                'turn_nr': turn_nr, 
                'prompt_type': 'pt-pt' if is_pt_pt else 'normal'  
               } | turn for turn_nr, turn in enumerate(turns)
            )

        scores = calculate_scores(classifiers, [ row['assistant'].split('</think>')[-1] for row in results ])
        responses = [
            row | score_1 | score_2 for row, score_1, score_2 in zip(results, *scores) 
        ]

        # out_filename = filename.rstrip('.json') + '_scored_v1.csv'
        frame = pd.DataFrame( data = responses )

        frame.set_index('prompt_id', inplace=True)
        frame.to_csv(out_filename)
        print(f">> storing results in '{out_filename}'")

if __name__ == '__main__': 
    main()

# srun --pty --gres=gpu:1 --mem 50G  python score.py pt-pt-eval