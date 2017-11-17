from collections import Counter
import numpy as np
import cPickle as cp
import pdb
from constants import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ============ File specific constants =======================#
HIST_FILE = "data/4gram_histogram.txt"
NUM_HIST = 50
# ============== BUILD THE VOCAB ================#
word_counter = Counter()
with open(TRAIN_FILE) as tf:
    for line in tf:
        line = line.lower().strip().split(' ')
        for word in line:
            word_counter[word] += 1

top_k_words = map(lambda x: x[0], word_counter.most_common(n=VOCAB_SIZE - 3))  # 3 special tokens
# Add <START>,<END>, <UNK>
word2ix = {START_TOK: 0, END_TOK: 1, UNK_TOK: 2}
for w in top_k_words:
    word2ix[w] = len(word2ix)
# Save the vocab
# cp.dump(word2ix, open(VOCAB_FILE, "wb"))
# ================ 4 gram histogram ==============#
# TODO: Plot the histogram
four_gram_count = {}
with open(TRAIN_FILE) as tf:
    for line in tf:
        line = line.lower().strip().split(' ')
        line = [START_TOK] + line + [END_TOK]
        for ix in xrange(len(line) - 4 + 1):
            four_gram = line[ix: ix + 4]
            four_gram = ' '.join([x if x in word2ix else "<UNK>" for x in four_gram])
            four_gram_count[four_gram] = dict.setdefault(four_gram_count, four_gram, 0) + 1
four_gram_count = sorted(four_gram_count.items(), key=lambda x: x[1], reverse=True)
with open(HIST_FILE, "wb") as hf:
    for ix in xrange(NUM_HIST):
        write_buf = "{}\t{}\n".format(four_gram_count[ix][0], four_gram_count[ix][1])
        hf.write(write_buf)
_, freq = map(list, zip(*four_gram_count))
xaxis = range(1, len(freq) + 1)
plt.xlabel('Four Grams')
plt.ylabel('Frequency')
plt.xlim(xmin=-1000, xmax=xaxis[-1] + 1)
fig, = plt.plot(xaxis, np.array(freq), label="fourgram_freq")
plt.savefig('Images_NGRAM/fourgram_histogram.png')
plt.close()
# =================================================#
