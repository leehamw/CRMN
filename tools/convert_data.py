#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: convert_data.py
"""

import json
import ast
import numpy as np
import re

from tools.misc import match_1, match_2

concat_token = '*'  # TODO
SILENCE = '<SILENCE>'
goal_split = '-->'
BOT_FIRST = 'Bot 主动'
USER_FIRST = 'User 主动'
TEST_RESPONSE = '<TEST_RESPONSE>'
TASK = '<PLACE_HOLD>'


def gen_task():
    return TASK  # TODO change to goal


def gen_goal(goal):
    goals = goal.split(goal_split)
    first_goal = goals[0]
    if BOT_FIRST in first_goal:
        assert USER_FIRST not in first_goal
        return True
    else:
        assert BOT_FIRST not in first_goal
        return False


def gen_uid(length):
    assert length % 2 == 0
    return list(map(lambda i: '0' if i % 2 != 0 else '1', list(range(length))))


def gen_dialog(conversation, bot_first=False, mode='train'):
    if mode != 'test':
        assert len(conversation) != 0

    new_conversation = [SILENCE] if bot_first else []
    for sent in conversation:
        tmp = re.sub('\[\d+\]', '', sent).strip()
        new_conversation.append(re.sub('\s+', ' ', tmp))

    if mode == 'test':
        assert len(new_conversation) % 2 != 0
        new_conversation.append(TEST_RESPONSE)
    else:
        if len(new_conversation) % 2 != 0:
            new_conversation.pop()

    assert len(new_conversation) % 2 == 0
    return new_conversation


def gen_kb(knowledge):
    new_knowledge, final_knowledge, objs = [], [], []
    for triple in knowledge:
        sub, rel, obj = triple
        # assert ' ' not in sub
        # assert ' ' not in rel
        # new_obj = obj.replace(' ', concat_token)

        sub = re.sub('\s+', concat_token, sub)
        rel = re.sub('\s+', concat_token, rel)
        obj = re.sub('\s+', concat_token, obj)
        objs.append(obj)

        new_triple = [sub, rel, obj]
        sub_triple = [sub, sub, sub]  # TODO

        if sub_triple not in new_knowledge:
            new_knowledge.append(sub_triple)
        if new_triple not in new_knowledge:
            new_knowledge.append(new_triple)

    for new_trip in new_knowledge:
        if new_trip[0] == new_trip[1] == new_trip[2] and new_trip[2] in objs:
            continue
        else:
            final_knowledge.append(new_trip)

    assert len(final_knowledge) != 0
    # return new_knowledge if len(new_knowledge) != 0 else ["<pad> <pad> <pad>"]
    return final_knowledge


def gen_others(dialog, kb):
    kb_index = []
    gold_entity = []
    for s in range(1, len(dialog), 2):
        gold, kb_ptr = [], []
        sys = dialog[s]
        for triple in kb:
            assert len(triple) == 3
            old_obj = triple[-1]
            new_obj = old_obj.replace(concat_token, ' ')

            score, sub_obj = match_2(new_obj, sys)  # todo modify match rules
            if score != -1:  # todo separate 'new_obj in sys' from it
                kb_ptr.append(1)
                assert sub_obj in sys
                dialog[s] = sys.replace(sub_obj, old_obj)
                gold.append(old_obj)
            else:
                assert sub_obj == ''
                kb_ptr.append(0)

        kb_index.append(kb_ptr)
        gold_entity.append(list(set(gold)))
    return kb_index, gold_entity


def gen_situation(situation):
    assert isinstance(situation, str)
    return re.sub('\s+', ' ', situation).strip()


def gen_user_profile(user_profile):
    assert isinstance(user_profile, dict)
    res = []
    for k, v in user_profile.items():
        assert isinstance(k, str)
        if isinstance(v, str):
            res.append(' '.join([k, v]))
        else:
            assert isinstance(v, list)
            res.append(' '.join([k, ' '.join(v)]))  # todo add concat_token
    return res


def convert_text_for_model(input_file, out_file, mode):
    all_samples = []
    max_response = -1
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            info = json.loads(line)
            situation = info['situation']
            goal = info['goal']
            user_profile = info['user_profile']
            knowledge = info['knowledge']
            conversation = info['conversation' if mode != 'test' else 'history']

            bot_first = gen_goal(goal)
            dialog = gen_dialog(conversation, bot_first, mode)
            kb = gen_kb(knowledge)
            kb_index, gold_entity = gen_others(dialog, kb)
            sample = {
                'task': gen_task(),
                'uid': gen_uid(len(dialog)),
                'dialog': dialog,
                'gold_entity': gold_entity,
                'kb_index': kb_index,
                'kb': list(map(lambda triple: ' '.join(triple), kb)),
                'situation': gen_situation(situation),
                'user_profile': gen_user_profile(user_profile)
            }

            all_samples.append(sample)

    print("total samples:", len(all_samples))  # 6618/946/4645

    for i, s in enumerate(all_samples):
        if len(s['uid']) == 0:
            print("index=%d utterance is None! filtered." % i)
            del all_samples[i]
        dialog = s['dialog']
        length = max([len(sent.split()) for i, sent in enumerate(dialog) if i % 2 == 0])
        max_response = max(max_response, length)

    print("max_response:", max_response)  # 68/64/48
    print("max utterances:", max([len(s['uid']) for s in all_samples]))      #
    print("min utterances:", min([len(s['uid']) for s in all_samples]))      #
    print("avg utterances:", np.mean([len(s['uid']) for s in all_samples]))  #
    print("max kb triples:", max([len(s['kb']) for s in all_samples]))       #
    print("min kb triples:", min([len(s['kb']) for s in all_samples]))       #
    print("avg kb triples:", np.mean([len(s['kb']) for s in all_samples]))   #
    with open(out_file, 'w') as fw:
        for sample in all_samples:
            line = json.dumps(sample)
            fw.write(line)
            fw.write('\n')


def compute_len(input_file):
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            info = json.loads(line)
            situation = info['user_profile']
            print(len(situation))


if __name__ == '__main__':
    in_data_dir = "../data/CRDATA"
    out_data_dir = "../data/CRDATA"
    modes = ['train', 'dev', 'test']

    for mode in modes:
        input_file = "%s/%s.txt" % (in_data_dir, mode)
        out_file = "%s/%s.data.txt" % (out_data_dir, mode)
        convert_text_for_model(input_file, out_file, mode)

    # for mode in modes:
    #     input_file = "%s/%s.data.txt" % (in_data_dir, mode)
    #     compute_len(input_file)
