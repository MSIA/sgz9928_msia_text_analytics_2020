import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os
from pathlib import Path
import collections
import pickle

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_file(filename, mypath='./speech_transcripts/Biden'):
    filepath = os.path.join(mypath, filename)
    text = Path(filepath).read_text()
    return text

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def model_input(data):
    for i in range(len(data)):
        yield gensim.models.doc2vec.TaggedDocument(data[i], [i])

if __name__ == "__main__":
    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'thank', 'you', 's'])

    # Read all speeches into one string
    list_of_files = os.listdir("./speech_transcripts/Biden")
    texts = ""
    for filename in list_of_files:
        try:
            text = read_file(filename)
            texts = texts + text + " "
        except UnicodeDecodeError as e:
            print(f'Skipping {filename} due to Unicode error')

    # Splitting each paragraph as a separate document
    #corpus = texts.split('\n')
    corpus = nltk.tokenize.sent_tokenize(texts)
    print(f"Total number of paragraphs in corpus = {len(corpus)}")

    # Storing corpus for model serving
    with open('./artifacts/biden_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)

    # Pre processing
    data_words = list(sent_to_words(corpus))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words)
    trigram = gensim.models.Phrases(bigram[data_words])  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Form Trigrams
    data_words_triigrams = make_trigrams(data_words_bigrams)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_triigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    # Saving to file for use in query based summarization
    with open('./artifacts/biden_lemmatized.pkl', 'wb') as f:
        pickle.dump(data_lemmatized, f)

    # Preparing corpus for Gensim doc2vec model
    train_corpus = list(model_input(data_lemmatized))

    # Doc2vec embeddings
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=70)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("./artifacts/biden_d2v.model") # saving model for future use

    # Assessing the embeddings by finding best match of all in corpus documents
    """
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    counter = collections.Counter(ranks)
    print(f"Training documents best match by doc2vec vectors: {counter}")
    """