import json
import multiprocessing
from string import punctuation
import collections
import numpy as np

file="yelp_academic_dataset_review.json"

f = open('dataset_stats.txt', 'w')

# read the first 500,000 yelp reviews
lines=open(file, encoding="utf8").readlines()[:500000]
print(f'Sample text:\n{lines[0]}', file=f)

# a very simple tokenizer that splits on white space and gets rid of some punctuation
def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(punctuation).lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

def process_yelp_line(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read and tokenize the text
    text=json_object['text']
    tokens=tokenize(text)
    # read the label and convert to an integer
    label=int(json_object['stars'])
    # return the tokens and the label
    return tokens, label

# distribute the processing across the machine cpus
pool=multiprocessing.Pool(multiprocessing.cpu_count())
result=pool.map(process_yelp_line, lines)
# "unzip" the (tokens, label) tuples to a list of lists of tokens, and a list of labels
texts, labels = zip(*result)

# Descriptive statistics
print(f'\nNumber of documents in corpus = {len(texts)}', file=f)
print(f'Unique labels = {collections.Counter(list(labels))}', file=f)
stars, counts = np.unique(labels, return_counts=True)
print(f'\nDistribution of labels:\n{stars} {counts/sum(counts)}', file=f)
text_length = [len(each) for each in texts]
print(f'\nAverage word length of documents = {np.mean(text_length)}', file=f)

f.close()
