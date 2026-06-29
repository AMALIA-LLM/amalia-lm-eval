import evaluate


def chrf_plus_score_fn(items):  # This is a passthrough function
    return items


def chrf_plus(predictions, references):
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions, references=references, word_order=2)
    return results['score'] / 100


def chrf_plus_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=preds, references=golds, word_order=2)
    # results: {'score': 82.87263732906315, 'char_order': 6, 'word_order': 2, 'beta': 2}
    return results['score']