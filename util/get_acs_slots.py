from data_handler import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from tqdm import tqdm

train, dev, test = create_splits("data/data_preprocessed.json", "data/ontology.json", slots=True)

def reformat(data):
    reformatted = []
    for dialog in data:
        dia_slots = []
        for turn in dialog:
            if len(turn) == 0:
                dia_slots.append("none")
                pass
            else:
                for label in turn:
                    if "reference" in label[0].lower() or "id" in label[0].lower() or "phone" in label[0].lower():
                        label[1] = label[0]
                    for token in label:
                        dia_slots.append(token)
        reformatted.append([" ".join(dia_slots), len(dia_slots)])
    return reformatted

slots = {}
slots['train'] = reformat(train)

print( len(reformat(train)))
slots['dev'] = reformat(dev)
slots['test'] =  reformat(test)
f = open("data/slot_info.json", "w")

json.dump(slots, f)