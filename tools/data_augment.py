import json


def convert_train_for_augment(input_file, out_file):
    new_samples, new_samples_num = [], 0
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            info = json.loads(line)

            task = info['task']
            uid = info['uid']
            dialog = info['dialog']
            gold_entity = info['gold_entity']
            kb_index = info['kb_index']
            kb = info['kb']

            turn_num = sum(list(map(lambda s: int(s), uid)))
            new_samples_num += turn_num

            for turn in range(turn_num):
                turn_odd, turn_even = turn + 1, turn * 2 + 2
                turn_uid = uid[:turn_even]
                turn_dialog = dialog[:turn_even]
                turn_gold_entity = gold_entity[:turn_odd]
                turn_kb_index = kb_index[:turn_odd]

                new_sample = {
                    'task': task,
                    'uid': turn_uid,
                    'dialog': turn_dialog,
                    'gold_entity': turn_gold_entity,
                    'kb_index': turn_kb_index,
                    'kb': kb
                }
                new_samples.append(new_sample)

    assert new_samples_num == len(new_samples)
    print("total new samples:", len(new_samples))

    with open(out_file, 'w') as fw:
        for sample in new_samples:
            line = json.dumps(sample)
            fw.write(line)
            fw.write('\n')


if __name__ == '__main__':
    data_dir = "../data/CRDATA"
    mode, new_mode = 'train', 'train_aug'

    input_file = "%s/%s.data.txt" % (data_dir, mode)
    output_file = "%s/%s.data.txt" % (data_dir, new_mode)
    convert_train_for_augment(input_file, output_file)