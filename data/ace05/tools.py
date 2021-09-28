import json

l = ["ace05_train.json", "ace05_dev.json", "ace05_test.json"]
for j in l:
    f = json.load(open(j))[:200]
    json.dump(f, open("debug_" + j, 'w'), indent=2, ensure_ascii=False)