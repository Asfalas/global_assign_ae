import json

l = ["ace05_train.json", "ace05_dev.json", "ace05_test.json"]
# for j in l:
#     f = json.load(open(j))[:50]
#     json.dump(f, open("debug_" + j, 'w'), indent=2, ensure_ascii=False)
data = json.load(open(l[0]))
lens = []
tlens = []
n = 0
t = 0
for d in data:
    args = d['args']
    lens.append(len(args))
    tmp = 0
    for arg in args:
        if arg['role'] == 'None':
            n += 1
        else:
            t += 1
            tmp += 1
    tlens.append(tmp)

lens = sorted(lens)
tlens = sorted(tlens)


print(n, t)
print(lens[int(len(lens) * 0.999)])
print(lens[int(len(lens) * 0.99)])
print(lens[int(len(lens) * 0.9)])
print(tlens[int(len(tlens) * 0.999)])
print(tlens[int(len(tlens) * 0.99)])
print(tlens[int(len(tlens) * 0.90)])