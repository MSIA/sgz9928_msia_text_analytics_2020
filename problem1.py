import os
from pathlib import Path
import time
import numpy as np
import re
import string
import tracemalloc

import nltk
from nltk.corpus import stopwords
from collections import Counter

# Reading in files from folder 'talk.religious.misc'
mypath = "/Users/shreyashiganguly/Documents/Northwestern_MSiA/Fall2020/Text_Analytics/HW1/20_newsgroups/talk.politics.guns"

def read_file(filename, mypath=mypath):
    filepath = os.path.join(mypath, filename)
    filesize = os.path.getsize(filepath)
    text = Path(filepath).read_text()
    return filesize, text

def normalize(text):
    # 1. Converting to lower case
    text = text.lower()
    # 2. Removing numbers
    text = re.sub(r'\d+', '', text)
    # 3. Removing punctuations
    text = text.translate(str.maketrans("","", string.punctuation))
    # 4. Removing trailing and leading whitespaces
    text = text.strip()
    return text

def text_tokenize(text, filename):
    text_lines = text.splitlines()
    stop_words = set(stopwords.words('english'))
    token_file = open(filename, 'w')
    for line in text_lines:
        nltk_words = nltk.word_tokenize(line)   # Converting each sentence into tokens
        nltk_words_clean = [i for i in nltk_words if i not in stop_words] # Removing stop words
        print(" ".join(nltk_words_clean), file=token_file) # writing to file
        print(Counter(nltk_words_clean))
    token_file.close()

if __name__ == '__main__':
    num_files = 0
    file_size = 0
    list_of_files = os.listdir(mypath)
    for filename in list_of_files:
        try:
            size, text = read_file(filename)
            num_files += 1
            file_size += size
            text_norm = normalize(text)
            text_tokenize(text_norm, 'processed_files/'+filename+'.txt')
        except UnicodeDecodeError as e:
            print(f'Skipping {filename} due to Unicode error')

    print(f"Number of files parsed = {num_files}")
    print(f"Total size of text corpus = {file_size} bytes")
