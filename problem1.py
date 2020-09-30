import os
from pathlib import Path
import time
import numpy as np
import tracemalloc

import nltk
from nltk.corpus import stopwords
import spacy
import stanza

# Reading in files from folder 'talk.religious.misc'
mypath = "/Users/shreyashiganguly/Documents/Northwestern_MSiA/Fall2020/Text_Analytics/HW1/20_newsgroups/talk.religion.misc"

def read_file(filename, mypath=mypath):
    filepath = os.path.join(mypath, filename)
    filesize = os.path.getsize(filepath)
    text = Path(filepath).read_text()
    return filesize, text

if __name__ == '__main__':
    f = open('nlp_library_comparison.txt', 'w')
    text_corpus = ""
    num_files = 0
    file_size = 0
    list_of_files = os.listdir(mypath)
    for filename in list_of_files:
        try:
            size, text = read_file(filename)
            text_corpus = text_corpus + text
            num_files += 1
            file_size += size
        except UnicodeDecodeError as e:
            print(f'Skipping {filename} due to Unicode error')

    print(f"Number of files parsed = {num_files}", file=f)
    print(f"Total size of text corpus = {file_size} bytes", file=f)
    
    # Test NLTK
    tracemalloc.start()
    t0 = time.time()
    nltk_sentences = nltk.sent_tokenize(text_corpus)    # sentence tokenization
    t1 = time.time()
    nltk_sentence_time = t1 - t0
    print(f'Sample sentences parsed:\n {nltk_sentences[0:5]}', file=f)
    print(f'NLTK sentence tokenizer runs in = {np.around(nltk_sentence_time,3)} seconds\n', file=f)

    t0 = time.time()
    nltk_words = nltk.word_tokenize(text_corpus)    # word tokenization
    t1 = time.time()
    nltk_words_time = t1 - t0
    print(f'Sample words parsed:\n {nltk_words[0:10]}', file=f)
    print(f'NLTK words tokenizer runs in = {np.around(nltk_words_time,3)} seconds\n', file=f)

    t0 = time.time()
    por_stem = nltk.PorterStemmer()
    nltk_stem_words = [por_stem.stem(word) for word in nltk_words]     # stemming
    t1 = time.time()
    nltk_stemming_time = t1 - t0
    print(f'Sample stem words parsed:\n {nltk_stem_words[10000:10010]}', file=f)
    print(f'NLTK stemming runs in = {np.around(nltk_stemming_time,3)} seconds\n', file=f)

    t0 = time.time()
    stop_words = set(stopwords.words('english')) 
    filtered_words = [w for w in nltk_stem_words if not w in stop_words]     # stop words removal
    t1 = time.time()
    nltk_stop_word_time = t1 - t0
    print(f'Number of words removed as stop words = {len(nltk_stem_words) - len(filtered_words)}', file=f)
    print(f'NLTK stop words removed in = {np.around(nltk_stop_word_time,3)} seconds\n', file=f)

    t0 = time.time()
    nltk_pos_tag = nltk.pos_tag(filtered_words)     # POS tagging
    t1 = time.time()
    nltk_pos_tag_time = t1 - t0
    print(f'Sample pos tagged tokens:\n {nltk_pos_tag[10000:10010]}', file=f)
    print(f'NLTK pos tagging runs in = {np.around(nltk_pos_tag_time,3)} seconds\n', file=f)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage with NLTK is {current / 10**6} MB; Peak was {peak / 10**6} MB\n", file=f)
    tracemalloc.stop()
    
    # Test spacy
    tracemalloc.start()
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3000000
    doc = nlp(text_corpus)
    
    spacy_tokens = []
    spacy_pos = []
    spacy_lemma = []

    t0 = time.time()
    for token in doc:
        spacy_tokens.append(token.text)
    t1 = time.time()
    spacy_token_time = t1 - t0
    print(f'Sample tokens:\n {spacy_tokens[10:20]}', file=f)
    print(f'Spacy tokenization runs in = {np.around(spacy_token_time,3)} seconds\n', file=f)

    t0 = time.time()
    for token in doc:
        spacy_pos.append(token.pos_)
    t1 = time.time()
    spacy_pos_time = t1 - t0
    print(f'Sample pos tags:\n {spacy_pos[10:20]}', file=f)
    print(f'Spacy tokenization runs in = {np.around(spacy_pos_time,3)} seconds\n', file=f)

    t0 = time.time()
    for token in doc:
        spacy_lemma.append(token.lemma_)
    t1 = time.time()
    spacy_lemma_time = t1 - t0
    print(f'Sample lemma:\n {spacy_lemma[10:20]}', file=f)
    print(f'Spacy lemmatization runs in = {np.around(spacy_lemma_time,3)} seconds\n', file=f)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage with Spacy is {current / 10**6} MB; Peak was {peak / 10**6} MB\n", file=f)
    tracemalloc.stop()
    

    # Test Stanza
    tracemalloc.start()
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
    doc = nlp(text_corpus)

    t0 = time.time()
    stanza_token = [word.text for sent in doc.sentences for word in sent.words]
    t1 = time.time()
    stanza_token_time = t1 - t0
    print(f'Sample tokens:\n {stanza_token[10:20]}', file=f)
    print(f'Stanza tokenization runs in = {np.around(stanza_token_time,3)} seconds\n', file=f)

    t0 = time.time()
    stanza_lemma = [word.lemma for sent in doc.sentences for word in sent.words]
    t1 = time.time()
    stanza_lemma_time = t1 - t0
    print(f'Sample lemma:\n {stanza_lemma[100:120]}', file=f)
    print(f'Stanza lemmatization runs in = {np.around(stanza_lemma_time,3)} seconds\n', file=f)

    t0 = time.time()
    stanza_pos = [word.xpos for sent in doc.sentences for word in sent.words]
    t1 = time.time()
    stanza_pos_time = t1 - t0
    print(f'Sample pos:\n {stanza_pos[100:120]}', file=f)
    print(f'Stanza POS tagging runs in = {np.around(stanza_pos_time,3)} seconds\n', file=f)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage with Stanza is {current / 10**6} MB; Peak was {peak / 10**6} MB\n", file=f)
    tracemalloc.stop()

    f.close()

