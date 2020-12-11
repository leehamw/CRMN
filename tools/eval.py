#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: eval.py
"""
import argparse
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from source.utils.metrics import moses_multi_bleu
from source.utils.metrics import compute_prf


def eval_f1(eval_fp):

    f1_scores = []

    with open(eval_fp, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            hyps = list(dialog["result"].replace('*', '').replace(' ', ''))
            refs = list(dialog["target"].replace('*', '').replace(' ', ''))

            hyp_list, ref_list = set(hyps), set(refs)
            common = [i for i in ref_list if i in hyp_list]
            f1_score = len(common) / len(ref_list)

            f1_scores.append(f1_score)

    f1 = np.mean(f1_scores)
    return f1


def eval_bleu_1_2(eval_fp):

    bleu_1_scores = []
    bleu_2_scores = []

    with open(eval_fp, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            hyp = dialog["result"].replace('*', ' ').split()
            ref = dialog["target"].replace('*', ' ').split()
            try:
                bleu_1 = sentence_bleu(references=[ref], hypothesis=hyp,
                                       smoothing_function=SmoothingFunction().method7,
                                       weights=[1, 0, 0, 0])
            except:
                bleu_1 = 0
            try:
                bleu_2 = sentence_bleu(references=[ref], hypothesis=hyp,
                                       smoothing_function=SmoothingFunction().method7,
                                       weights=[0.5, 0.5, 0, 0])
            except:
                bleu_2 = 0

            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)

    assert len(bleu_1_scores) == len(bleu_2_scores)
    bleu_1_score = np.mean(bleu_1_scores)
    bleu_2_score = np.mean(bleu_2_scores)

    # bleu_score = (bleu_1_score + bleu_2_score) / 2
    return bleu_1_score, bleu_2_score


def eval_bleu(eval_fp):
    hyps = []
    refs = []
    with open(eval_fp, 'r') as fr:
        for line in fr:
            dialog = json.loads(line.strip())
            pred_str = dialog["result"]
            gold_str = dialog["target"]
            hyps.append(pred_str)
            refs.append(gold_str)
    assert len(hyps) == len(refs)
    hyp_arrys = np.array(hyps)
    ref_arrys = np.array(refs)

    bleu_score = moses_multi_bleu(hyp_arrys, ref_arrys, lowercase=True)

    return bleu_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--eval_dir", type=str)
    args = parser.parse_args()

    mode = args.mode
    assert mode == 'valid'

    data_dir = args.data_dir
    eval_dir = args.eval_dir
    eval_file = "%s/output.txt" % eval_dir

    # todo delete <unk>

    # cal f1
    f1 = eval_f1(eval_file)
    output_str = "F1 SCORE: %.2f%%\n" % (f1 * 100)

    # cal bleu
    bleu = eval_bleu(eval_file)
    bleu_1, bleu_2 = eval_bleu_1_2(eval_file)
    output_str += "BLEU_MOSE SCORE: %.3f\n" % bleu
    output_str += "BLEU_1 SCORE: %.3f\n" % bleu_1
    output_str += "BLEU_2 SCORE: %.3f" % bleu_2

    out_file = "%s/eval.result.txt" % eval_dir
    with open(out_file, 'w') as fw:
        fw.write(output_str)
    print("Saved evaluation results to '{}.'".format(out_file))
