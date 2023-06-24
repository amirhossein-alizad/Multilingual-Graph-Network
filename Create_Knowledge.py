import json

data = {}

with open('dict.txt', 'r') as file:
    for line in file:
        line = line.strip().replace(']', '')
        line = line.split('[')
        word = line[0][:-1]
        senses = line[1:][0].split()
        data[word] = senses

data = {key: [v.strip("',") for v in value] for key, value in data.items()}

with open('knowledge.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
