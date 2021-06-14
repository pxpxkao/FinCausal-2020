import numpy as np
#label2id= {"_":0, "C":1, "E":2}
label2id = {"_":0, "B-C":1, "I-C":2, "B-E":3, "I-E":4}
def readfile(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            data.append([label2id[tag] for tag in line])
    return data

data = readfile('task2.train.tgt')
data = np.array(data)
print('Shape of data:', data.shape)
print('Example:\n', data[0])

num_labels = 5
cnt_trans = np.zeros((num_labels, num_labels))
#print(cnt_trans)
for t in range(len(data)):
    for i in range(len(data[t])-1):
        prev_, now_ = data[t][i], data[t][i+1]
        cnt_trans[prev_, now_] += 1

trans = np.log((cnt_trans.T / cnt_trans.sum(1)).T)
print('Transition Matrix:\n', trans)