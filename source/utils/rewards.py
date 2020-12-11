#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
File: source/utils/rewards.py
"""

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from source.utils.metrics import moses_multi_bleu, compute_prf
from source.utils.misc import Pack


def reward_fn1(self, preds, targets, gold_ents, ptr_index, task_label):
    """
    reward_fn1
    General reward
    """
    # parameters
    alpha1 = 1.0
    alpha2 = 0.3

    # acc reward
    '''
    # get the weighted mask
    no_padding_mask = preds.ne(self.padding_idx).float()
    trues = (preds == targets).float()
    if self.padding_idx is not None:
        weights = no_padding_mask
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    '''

    pred_text = self.tgt_field.denumericalize(preds)
    tgt_text = self.tgt_field.denumericalize(targets)
    batch_size = targets.size(0)
    batch_kb_inputs = self.kbs[:batch_size, :, :]
    kb_plain = self.kb_field.denumericalize(batch_kb_inputs)

    result = Pack()
    result.add(pred_text=pred_text, tgt_text=tgt_text, gold_ents=gold_ents, kb_plain=kb_plain)
    result_list = result.flatten()

    # bleu reward
    bleu_score = []
    for res in result_list:
        hyp_toks = res.pred_text.split()
        ref_toks = res.tgt_text.split()
        try:
            bleu_1 = sentence_bleu(references=[ref_toks], hypothesis=hyp_toks,
                                   smoothing_function=SmoothingFunction().method7,
                                   weights=[1, 0, 0, 0])
        except:
            bleu_1 = 0
        try:
            bleu_2 = sentence_bleu(references=[ref_toks], hypothesis=hyp_toks,
                                   smoothing_function=SmoothingFunction().method7,
                                   weights=[0.5, 0.5, 0, 0])
        except:
            bleu_2 = 0
        bleu = (bleu_1 + bleu_2) / 2
        bleu_score.append(bleu)
    bleu_score = torch.tensor(bleu_score, dtype=torch.float)

    # entity f1 reward
    f1_score = []
    report_f1 = []
    for res in result_list:
        if len(res.gold_ents) == 0:
            f1_pred = 1.0
        else:
            # TODO: change the way
            #gold_entity = ' '.join(res.gold_ents).replace('_', ' ').split()
            #pred_sent = res.pred_text.replace('_', ' ')
            gold_entity = res.gold_ents
            pred_sent = res.pred_text
            f1_pred, _ = compute_prf(gold_entity, pred_sent,
                                     global_entity_list=[], kb_plain=res.kb_plain)
            report_f1.append(f1_pred)
        f1_score.append(f1_pred)
    if len(report_f1) == 0:
        report_f1.append(0.0)
    f1_score = torch.tensor(f1_score, dtype=torch.float)
    report_f1 = torch.tensor(report_f1, dtype=torch.float)

    if self.use_gpu:
        bleu_score = bleu_score.cuda()
        f1_score = f1_score.cuda()
        report_f1 = report_f1.cuda()

    # compound reward
    #reward = alpha1 * bleu_score.unsqueeze(-1) + alpha2 * f1_score.unsqueeze(-1)
    reward = alpha1 * bleu_score.unsqueeze(-1)

    return reward, bleu_score, report_f1
