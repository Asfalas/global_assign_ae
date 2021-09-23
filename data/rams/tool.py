import json
import sys
from tqdm import tqdm

source_file_list = ['train.json', 'dev.json', 'test.json']
target_file_list = ['ace05_train.json', 'ace05_dev.json', 'ace05_test.json']

for s, t in zip(source_file_list, target_file_list):
    data = json.load(open(s))
    new_data = []
    for d in tqdm(data):
        if d['event_type'] == 'None':
            continue
        tmp = {}
        tmp['tokens'] = d['tokens']
        tmp['evt_type'] = d['event_type']
        tmp['evt_mention'] = d['trigger_tokens']
        tmp['evt_beg'] = d['trigger_start']
        tmp['evt_end'] = d['trigger_end']
        assert d['tokens'][tmp['evt_beg']: tmp['evt_end']+1] == tmp['evt_mention']
        tmp['args'] = []
        for ent in d['entities']:
            tent = {}
            tent['arg_mention'] = ent['tokens']
            tent['arg_beg'] = ent['idx_start']
            tent['arg_end'] = ent['idx_end']
            tent['role'] = ent['role']
            assert d['tokens'][tent['arg_beg']: tent['arg_end']+1] == ent['tokens']
            tmp['args'].append(tent)
        new_data.append(tmp)

    json.dump(new_data, open(t, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(len(new_data))



data = json.load(open('ace05_dev.json'))
for d in tqdm(data):
    json.dump(d, sys.stdout, indent=2, ensure_ascii=False)
    break