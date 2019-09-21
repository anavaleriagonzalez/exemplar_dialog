
import numpy as np 
import pandas as pd 

with open("results/groundtruth-final.txt") as f:
    y_true = f.readlines()

with open("results/preds-final.txt") as f:
    y_hat = f.readlines()

with open("results/baseline_groundtruth.txt") as f:
    b_y = f.readlines()

with open("results/baseline_model_preds.txt") as f:
    b_yhat = f.readlines()

#rearrange baseline preds
baseline_preds = []
for i in range(len(b_y)):
    idx = b_y.index(y_true[i])
    baseline_preds.append(b_yhat[idx])



print("loading embeddings file...")
w2v = {}
with open("data/glove.6B.50d.txt", 'r') as fp:
    f = fp.readlines()
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) > 51:
            continue
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs


dim = 50 # dimension of embeddings

scores = {}
scores_base = {}


for i in range(len(y_true)):
    tokens1 = y_true[i].strip()[2:-5].split(" ")
    
    tokens2 = y_hat[i].strip().split(" ")

    tokens_baseline = baseline_preds[i].strip().split(" ")

    X= np.zeros((dim,))
    for tok in tokens1:
        if tok in w2v:
            X+=w2v[tok]

    Y = np.zeros((dim,))
    for tok in tokens2:
        if tok in w2v:
            Y += w2v[tok]

    base = np.zeros((dim,))
    for tok in tokens_baseline:
        if tok in w2v:
            base += w2v[tok]

    # if none of the words in ground truth have embeddings, skip
    if np.linalg.norm(X) < 0.00000000001:
        continue

    # if none of the words have embeddings in response, count result as zero
    if np.linalg.norm(Y) < 0.00000000001:
        scores[i] = o
        continue

    if np.linalg.norm(base) < 0.00000000001:
        scores_base[i]= o_base
        continue

    X = np.array(X)/np.linalg.norm(X)
    Y = np.array(Y)/np.linalg.norm(Y)

    base = np.array(base)/np.linalg.norm(base)
    o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

    o_base = np.dot(X, base.T)/np.linalg.norm(X)/np.linalg.norm(base)

    scores[i] = o
    scores_base[i]= o_base

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

f = open("results/improvements.txt", "w")
#print the top 20 and bottom 20 sentences
f.write("Top scored sentences....\n")
for i in range(100):
    f.write("score: " + str(sorted_scores[i][1]) + "\ntruth: "+ y_true[sorted_scores[i][0]]+"\npred: " + y_hat[sorted_scores[i][0]])
    f.write("\n\n")
    f.write("score_base: "+ str(scores_base[sorted_scores[i][0]]) + "\ntruth: " + y_true[sorted_scores[i][0]] + "\npred: " + baseline_preds[sorted_scores[i][0]])
    f.write("\n--------------------\n")

f.write("*********************************************")
f.write("Lowest scored sentences...")
for i in range(100):
    f.write("score: " + str(sorted_scores[i-20][1]) + "\ntruth: "+ y_true[sorted_scores[i-20][0]]+"\npred: " + y_hat[sorted_scores[i-20][0]])
    f.write("\n\n")
    f.write("score_base: "+ str(scores_base[sorted_scores[i-20][0]]) + "\ntruth: " + y_true[sorted_scores[i-20][0]] + "\npred: " + baseline_preds[sorted_scores[i-20][0]])
    f.write("\n--------------------\n")

f.close()
'''
#######################################
score:  0.9992900767002726 truth:  ('<sou> your booking was successful and your reference number is hotel-reference . is there anything else i can help you with ? <eou>', 0)
 pred:  <sou> your booking was successful . your reference number is hotel-reference . is there anything else i can help with ?

score_base:  0.9320724506096328 truth:  ('<sou> your booking was successful and your reference number is hotel-reference . is there anything else i can help you with ? <eou>', 0)
 pred:  <sou> the booking was successful . the table will be reserved for 15 minutes . reference number is : restaurant-reference .

#######################################

score:  0.9992283344007203 truth:  ("<sou> you 're welcome . can i help you with anything else today ? <eou>", 0)
 pred:  <sou> you 're welcome . can i help with anything else ? <eou>

score_base:  0.9784617621216077 truth:  ("<sou> you 're welcome . can i help you with anything else today ? <eou>", 0)
 pred:  <sou> i have a listing for you . do you need free parking ? <eou>

#######################################

score:  0.9991189703684435 truth:  ('<sou> the reference number is restaurant-reference . is there anything else i can help you with ? <eou>', 0)
 pred:  <sou> sure thing . the reference number is restaurant-reference . is there anything else i can help you with ? <eou>

score_base:  0.8582320442994101 truth:  ('<sou> the reference number is restaurant-reference . is there anything else i can help you with ? <eou>', 0)
pred:  <sou> the address is cambridge leisure park , clifton way . <eou>

#######################################

'''
