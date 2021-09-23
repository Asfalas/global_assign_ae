import json
import sys
from tqdm import tqdm
import re
from _functools import reduce

def get_role(role):
    pattern = r'evt\d+arg\d+([a-zA-z]+)'
    m = re.findall(pattern, role)
    if m:
        return m[0]
    else:
        raise Exception('Invalid')

m = {}

source_file_list = ['train.jsonlines', 'dev.jsonlines', 'test.jsonlines']
target_file_list = ['rams_train.json', 'rams_dev.json', 'rams_test.json']

for s, t in zip(source_file_list, target_file_list):
    new_data = []
    sent_ent_map = {}
    for line in tqdm([line for line in open(s).readlines()]):
        d = json.loads(line)
        tokens = reduce(lambda x, y: x + y, d['sentences'])
        sentence = '@#@'.join(tokens)
        if sentence not in sent_ent_map:
            sent_ent_map[sentence] = set()
        for arg in d['gold_evt_links']:
            sent_ent_map[sentence].add('@#@'.join([str(i) for i in arg[1]]))
    
    for line in tqdm([line for line in open(s).readlines()]):
        d = json.loads(line)
        tokens = reduce(lambda x, y: x + y, d['sentences'])
        sentence = '@#@'.join(tokens)
        tmp = {}
        tmp['tokens'] = tokens
        tmp['evt_type'] = d['evt_triggers'][0][2][0][0]
        tmp['evt_beg'] = d['evt_triggers'][0][0]
        tmp['evt_end'] = d['evt_triggers'][0][1]
        tmp['evt_mention'] = tokens[tmp['evt_beg']: tmp['evt_end']+1]
        tmp['args'] = []
        arg_map = {}
        for arg in d['gold_evt_links']:
            tmp_arg = {}
            tmp_arg['arg_beg'] = arg[1][0]
            tmp_arg['arg_end'] = arg[1][1]
            tmp_arg['arg_mention'] = tokens[arg[1][0]: arg[1][1] + 1]
            tmp_arg['role'] = get_role(arg[2])
            arg_map['@#@'.join([str(arg[1][0]), str(arg[1][1])])] = tmp_arg

        for offsets in sent_ent_map[sentence]:
            if offsets in arg_map:
                tmp['args'].append(arg_map[offsets])
            else:
                beg = int(offsets.split('@#@')[0])
                end = int(offsets.split('@#@')[1])
                tmp_arg = {}
                tmp_arg['arg_beg'] = beg
                tmp_arg['arg_end'] = end
                tmp_arg['arg_mention'] = tokens[beg: end + 1]
                tmp_arg['role'] = 'None'
                tmp['args'].append(tmp_arg)
        # max_ent_len = max_ent_len if max_ent_len > len(tmp['args']) else len(tmp['args'])
        if len(tmp['args']) not in m:
            m[len(tmp['args'])] = 0
        m[len(tmp['args'])] += 1
        new_data.append(tmp)
    
    json.dump(new_data, open(t, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(len(new_data))

for k, v in m.items():
    print(k, v)
# data = json.load(open('rams_test.json'))
# for d in tqdm(data[1:]):
#     json.dump(d, sys.stdout, indent=2, ensure_ascii=False)
#     break