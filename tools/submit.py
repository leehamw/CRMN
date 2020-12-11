import argparse
import json
import re

from tools.convert_data import TEST_RESPONSE


def post_process(result):
    line = result.replace('*', ' ').replace('<unk>', '').strip()
    reply = re.sub('\s+', ' ', line)
    return reply


def gen_submit(input_file1, input_file2, out_file):

    vetify_sents = []
    with open(input_file1, 'r') as fr1:
        for line in fr1:
            info = json.loads(line)
            dialog = info['dialog']
            assert dialog[-1] == TEST_RESPONSE
            if len(dialog) == 2:
                vetify_sents.append(dialog[-2])
            else:
                vetify_sents.append(' '.join([dialog[-3], dialog[-2]]))

    with open(input_file2, 'r') as fr2, open(out_file, 'w', encoding='utf-8') as fw:
        for i, line in enumerate(fr2):
            info = json.loads(line)
            source = info['source']
            result = info['result']
            target = info['target']
            vetify_sent = vetify_sents[i]

            if '<unk>' not in source:
                assert vetify_sent == source, vetify_sent + '\n' + source
            assert target == TEST_RESPONSE

            reply = post_process(result)

            fw.write(reply)
            fw.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--eval_dir", type=str)
    args = parser.parse_args()

    mode = args.mode
    assert mode == 'test'

    in_data_dir = args.data_dir
    out_data_dir = args.eval_dir

    input_file1 = "%s/%s.data.txt" % (in_data_dir, mode)
    input_file2 = "%s/output.txt" % (out_data_dir)
    out_file = "%s/submit.txt" % (out_data_dir)

    gen_submit(input_file1, input_file2, out_file)
