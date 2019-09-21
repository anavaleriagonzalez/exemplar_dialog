from data_handler import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from tqdm import tqdm

train, dev, test = create_splits("data/data_preprocessed_acts.json", "data/ontology.json", slots=True)


def reformat(data):
    vocab_slots = []
    reformatted = []
    for dialog in data:
        dia_slots = []
        for i, turn in enumerate(dialog):

            if len(turn) == 0:
                dia_slots.append("none")
                vocab_slots.append("none")
                pass
            else:
                turn_reps = []
                for label in turn:
                    if type(label) == list:
                        l = " ".join(label[0].split('-'))
                        if "pricerange" in l:
                            l = l.replace("pricerange", "price range")
                        turn_reps.append(l)

                    elif type(label) == dict:
                        if i == 2:
                            for key in label.keys():
                                turn_reps.append([" ".join(key.split("-")).lower()])



                        else:
                            for key in label.keys():
                                turn_reps.append(" ".join(key.split("-")).lower())
                                for slot, value in label[key]:
                                    if slot=="none":
                                        pass
                                    else:
                                        turn_reps.append(slot.lower())
                                        if "id" in slot.lower() or "ref" in slot.lower() or "phone" in slot.lower():
                                            pass
                                        else:
                                            turn_reps.append(value)
                    else:
                        turn_reps.append("none")
                if i == 2:
                    dia_slots.append(turn_reps)
                    for list_ in turn_reps:
                        for word in " ".join(list_).lower().split():
                            vocab_slots.append(word)
                else:
                    dia_slots.append(" ".join(turn_reps).lower())
                    for word in " ".join(turn_reps).lower().split():
                        vocab_slots.append(word)
        reformatted.append([dia_slots, len(dia_slots[0]), len(dia_slots[1]),len(dia_slots[2] )])
    return reformatted, vocab_slots

slots = {}
slots['train'], vocab1 = reformat(train)

slots['dev'], vocab2 = reformat(dev)
slots['test'], vocab3 =  reformat(test)
f = open("data/slot_info.json", "w")


vocab = list(set(vocab1 +vocab2 + vocab3))

with open("data/vocab_acts.txt", "w") as voc:
    for word in vocab:
        print(word)
        voc.write(word.lower())
        voc.write("\n")


json.dump(slots, f)