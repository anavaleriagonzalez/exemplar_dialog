import json
import numpy as np
import random
from nltk.tokenize import sent_tokenize, word_tokenize
np.random.seed(123)
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load(path):

    with open(path, "r") as f:
        data = json.load(f)
    return data

def domain_specific(data,ontology_path, slots=False, delex=True):

    if slots == True:
        slot_info = []
        for i, dialog in enumerate(data):
            
            dialog_slots = []
            for turn in dialog['dialogue']:
            
                if turn['system_transcript'] == "":
                    dialog_slots.append(turn['system_acts'])
            
                    dialog_slots.append(turn['turn_label'])

                if turn['transcript'] == "":
                
                    dialog_slots.append(turn['system_acts'])

                elif len(turn['system_transcript'])> 1:
                    
                    dialog_slots.append(turn['system_acts'])

                    dialog_slots.append(turn['system_acts'])

                    dialog_slots.append(turn['turn_label'])

            
            dialog_chunks = list(chunks(dialog_slots, 3))

            
            for chunk in dialog_chunks: 
                slot_info.append(chunk)


        return slot_info
                
            

    ont = get_delex(ontology_path)

    reformatted = {}
    all_domains = []
    for i, dialog in enumerate(data):
        flattened_dialog = []
        dialog_slots = []
        for turn in dialog['dialogue']:
            
            if turn['system_transcript'] == "":
                flattened_dialog.append("soc")

                words = [word for sent in sent_tokenize(turn['transcript']) for word in word_tokenize(sent)]
                delex = []
                for word in words:
                    if word.lower() in ont.keys():
                        delex.append(ont[word.lower()])
                    else:
                        delex.append(word.lower())
                flattened_dialog.append( "<sou> " +" ".join(delex)+ " <eou>" )

            if turn['transcript'] == "":
                
                words = [word for sent in sent_tokenize(turn['system_transcript']) for word in word_tokenize(sent)]

                delex = []
                for word in words:
                    if word.lower() in ont.keys():
                        delex.append(ont[word.lower()])
                    else:
                        delex.append(word.lower())
                flattened_dialog.append( "<sou> " +" ".join(delex)+ " <eou>" )

                

            elif len(turn['system_transcript'])> 1:
                words = [word for sent in sent_tokenize(turn['system_transcript']) for word in word_tokenize(sent)]

                delex = []
                for word in words:
                    if word.lower() in ont.keys():
                        delex.append(ont[word.lower()])
                    else:
                        delex.append(word.lower())
                flattened_dialog.append(  "<sou> " +" ".join(delex)+ " <eou>")
                flattened_dialog.append(  "<sou> " +" ".join(delex)+ " <eou>" )

                words = [word for sent in sent_tokenize(turn['transcript']) for word in word_tokenize(sent)]
                delex = []
                for word in words:
                    if word.lower() in ont.keys():
                        delex.append(ont[word.lower()])
                    else:
                        delex.append(word.lower())
                flattened_dialog.append( "<sou> " +" ".join(delex)+ " <eou>")

            

        dialog_chunks = list(chunks(flattened_dialog, 3))

        for chunk in dialog_chunks:
            all_domains.append([chunk, len(chunk[0].split()), len(chunk[1].split()), len(chunk[2].split())])
  
    return all_domains

def create_splits(data_path, ontology_path, splits=[0.8,0.1], slots=False):
    data = load(data_path)
    np.random.shuffle(data)

    train_idx = int(len(data)*splits[0])
    dev_idx = int(len(data)*splits[1])

    train = data[0:train_idx]
    dev = data[train_idx:train_idx+dev_idx]
    test = data[train_idx+dev_idx::]

    train = domain_specific(train, ontology_path, slots)
    dev = domain_specific(dev, ontology_path, slots)
    test = domain_specific(test, ontology_path,  slots)
    return train, dev,  test


#data = read_data("data/data_preprocessed.json")
def get_vocab(data):
    vocab = set()
    
    for context in data:       
        for utt in context[0]:
            for word in utt.split():
                vocab.add(word)
               
    f = open('data/vocab.txt', 'w')
    for item in list(vocab):
        f.write(item+"\n")
    f.close()

def get_delex(path):
    ontology = load(path)

    new_ont = {}
    for key in ontology['values'].keys():
        if "reference" in key or "id" in key or "phone" in key:
            for value in ontology['values'][key]:
                new_ont[value] = key
    return new_ont

