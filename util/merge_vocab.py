f1 = open("data/vocab.txt")
v1 = f1.readlines()
f1.close()

f2 = open("data/vocab_acts.txt")
v2 = f2.readlines()
f2.close()

v = list(set(v1 + v2))

with open("data/vocab_final.txt", "w") as f:
    for word in v:
        f.write(word)

