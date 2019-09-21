import json

with open('data/data_preprocessed.json') as f:
    prepro = json.load(f)

with open('data/raw/data.json') as f:
    raw = json.load(f)

with open('data/raw/dialogue_acts.json') as f:
    acts = json.load(f)

#need to get the dialogue acts from a separate file
utt2file = {}

convos = []
for key in  raw.keys():
    utts = []
    for i in range(len(raw[key]['log'])):
        utts.append(raw[key]['log'][i]['text'].lower()) 

    convos.append([" ".join(utts), key])
    utt2file[" ".join(utts)] = key

trimmed = {}
for i, key in convos:
    if len(i) < 40:
        trimmed[i] = key
    else:
        trimmed[i[0:40]]=key

#getting file names for dialogue in preprocessed.json
files = []
for dialog in prepro:
    utts = []
    for turn in dialog['dialogue']:
      
        utts.append(turn["system_transcript"])
        utts.append(turn['transcript'])
    conv = " ".join(utts[1:-1]).lower()

    if conv in utt2file.keys():
        files.append(utt2file[conv].split(".")[0])

    else:
       
        if conv[0:40] in trimmed.keys():
            files.append(trimmed[conv[0:40]].split(".")[0])


for i in range(len(prepro)):    # for dialog in data
    for j in range(len(prepro[i]["dialogue"])):     # for turn in dialog
     
        if str(j) in acts[files[i]].keys():
            
            prepro[i]["dialogue"][j]['system_acts'] = [acts[files[i]][str(j)]]
    

with open('data/data_preprocessed_acts.json', "w") as f:
    json.dump(prepro,f)
