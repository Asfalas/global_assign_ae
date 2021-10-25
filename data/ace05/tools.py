import json
import sys

l = ["ace05_train.json", "ace05_dev.json", "ace05_test.json"]
for j in l:
    f = json.load(open(j))[:50]
    json.dump(f, open("debug_" + j, 'w'), indent=2, ensure_ascii=False)
data = json.load(open(l[0]))
ent_lens = []
arg_lens = []
for d in data:
    args = d['args']
    all_none = True
    ent_lens.append(len(args))
    tmp = 0
    for arg in args:
        if arg['role'] != 'None':
            tmp+=1
    arg_lens.append(tmp)

ent_lens = sorted(ent_lens)
arg_lens = sorted(arg_lens)
print(ent_lens[int(len(ent_lens) * 0.99)])
print(ent_lens[int(len(ent_lens) * 0.95)])
print(arg_lens[int(len(arg_lens) * 0.99)])
print(arg_lens[int(len(arg_lens) * 0.95)])

# input_files = ["train.json", "dev.json", "test.json"]
# output_files = ["ace05_train_1.json", "ace05_dev_1.json", "ace05_test_1.json"]

# for f, o in zip(input_files, output_files):
#     f = json.load(open(f))
#     json.dump(f[1], sys.stdout, indent=2, ensure_ascii=False)
#     break