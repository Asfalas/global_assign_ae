import json 
data = json.load(open("config.json"))
event_schema = data["event_schema"]
roles = set()
for k, v in event_schema.items():
    for i in v:
        roles.add(i)

data["role_list"] = list(roles)
json.dump(data, open("config.json", "w"), indent=2)