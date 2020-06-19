import sys
import json
import re
import string
from collections import Counter
import pickle
from copy import copy, deepcopy


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no',
                                 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no',
                                   'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def calculate_metrics(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall


def update_metrics(metrics, em, f1, prec, recall):
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall


def calculate_sp(prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, f1, prec, recall


def update_spmetrics(metrics, em, f1, prec, recall):
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall

def update_jointmetric(metrics, joint_em, joint_f1, joint_prec, joint_recall):
    metrics['joint_em'] += joint_em
    metrics['joint_f1'] += joint_f1
    metrics['joint_prec'] += joint_prec
    metrics['joint_recall'] += joint_recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        answer_prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    sp_prediction = None
    default_metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
                       'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
                       'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0, 'count': 0}
    complete_metrics = {}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        metric_keys = ["all", dp['type'], dp['level'], str(len(dp['supporting_facts'])) + "hop"]
        for metric_key in metric_keys:
            if metric_key not in complete_metrics:
                complete_metrics[metric_key] = deepcopy(default_metrics)
            complete_metrics[metric_key]["count"] += 1
        if cur_id not in answer_prediction:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, f1, prec, recall = calculate_metrics(answer_prediction[cur_id], dp['answer'])
            for metric_key in metric_keys:
                update_metrics(complete_metrics[metric_key], em, f1, prec, recall)
        if sp_prediction is not None:
            if cur_id not in sp_prediction:
                print('missing sp fact {}'.format(cur_id))
                can_eval_joint = False
            else:
                sp_em, sp_f1, sp_prec, sp_recall = calculate_sp(
                    sp_prediction[cur_id], dp['supporting_facts'])
                for metric_key in metric_keys:
                    update_spmetrics(complete_metrics[metric_key], sp_em, sp_f1, sp_prec, sp_recall)
        else:
            can_eval_joint = False

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em
            for metric_key in metric_keys:
                update_jointmetric(complete_metrics[metric_key], joint_em, joint_f1, joint_prec, joint_recall)

    for metric_key in complete_metrics.keys():
        metrics = complete_metrics[metric_key]
        for k in metrics.keys():
            if k != "count" and metrics["count"] > 0:
                metrics[k] /= metrics["count"]

    print(json.dumps(complete_metrics, indent=2))


if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])
