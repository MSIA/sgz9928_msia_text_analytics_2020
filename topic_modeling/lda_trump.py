import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os
from pathlib import Path
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

def read_file(filename, mypath='./speech_transcripts/Trump'):
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

if __name__ == "__main__":
    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'thank', 'you', 's'])

    # Read all speeches into one string
    list_of_files = os.listdir("./speech_transcripts/Trump")
    texts = ""
    for filename in list_of_files:
        try:
            text = read_file(filename)
            texts = texts + text + " "
        except UnicodeDecodeError as e:
            print(f'Skipping {filename} due to Unicode error')

    # Combining 50 sentences into one document - paragraph
    sentences = nltk.tokenize.sent_tokenize(texts)
    print(f"Total number of sentences in corpus = {len(sentences)}")

    corpus = []
    i = 0
    while(i < len(sentences)):
        paragraph = ' '.join(sentences[i:i+50])
        corpus.append(paragraph)
        i = i+50

    print(f"Total number of paragraphs in corpus = {len(corpus)}")

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

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    id2word.filter_extremes()

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    tfidf_model = gensim.models.TfidfModel(dictionary = id2word)
    corpus = [id2word.doc2bow(x) for x in texts]
    corpus_tfidf = tfidf_model[corpus]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf,
                                            id2word=id2word,
                                            num_topics=5,
                                            passes=20,
                                            random_state=100)

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus_tfidf))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, id2word, R=20)
    pyLDAvis.save_html(vis, './artifacts/Trump_lda.html')


