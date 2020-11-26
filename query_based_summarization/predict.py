import os, pickle
import networkx as nx
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

TRUMP_CORPUS = os.environ.get('TRUMP_CORPUS', "./artifacts/trump_corpus.pkl")
TRUMP_LEMMATIZED = os.environ.get('TRUMP_LEMMATIZED', "./artifacts/trump_lemmatized.pkl")
TRUMP_D2V_MODEL = os.environ.get('TRUMP_D2V_MODEL', "./artifacts/trump_d2v.model")

BIDEN_CORPUS = os.environ.get('BIDEN_CORPUS', "./artifacts/biden_corpus.pkl")
BIDEN_LEMMATIZED = os.environ.get('BIDEN_LEMMATIZED', "./artifacts/biden_lemmatized.pkl")
BIDEN_D2V_MODEL = os.environ.get('BIDEN_D2V_MODEL', "./artifacts/biden_d2v.model")
GLOVE_EMBEDDINGS_50D = os.environ.get('GLOVE_EMBEDDINGS_50D', "./artifacts/glove.6B.50d.txt")

# Loading model and artifacts
trump_d2v = Doc2Vec.load(TRUMP_D2V_MODEL)
biden_d2v = Doc2Vec.load(BIDEN_D2V_MODEL)

with open(TRUMP_CORPUS, 'rb') as f:
    trump_corpus = pickle.load(f)
with open(BIDEN_CORPUS, 'rb') as f:
    biden_corpus = pickle.load(f)
with open(TRUMP_LEMMATIZED, 'rb') as f:
    trump_lemma = pickle.load(f)
with open(BIDEN_LEMMATIZED, 'rb') as f:
    biden_lemma = pickle.load(f)

# Loading GloVe word embeddings
word_embeddings = {}
f = open(GLOVE_EMBEDDINGS_50D, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


def tokenize(text):
    """Splits user provided keywords into tokens"""
    tokens = map(lambda x: x.strip(',.&').lower(), text.split())
    return list(tokens)

def query_most_similar(query_token, model, corpus, data_lemmatized):
    inferred_vector = model.infer_vector(query_token)
    sims = model.docvecs.most_similar([inferred_vector], topn=100)

    # Merging relevant paragraphs into one 
    summ_lemmatized = []
    para = ""
    for index in range(len(sims)):
        para = para + corpus[sims[index][0]] + " "
        summ_lemmatized.append(data_lemmatized[sims[index][0]])

    summ_sent = nltk.tokenize.sent_tokenize(para)

    ## Using GloVe embeddings - averaged for each sentence
    sentence_vectors = []
    for sent in summ_lemmatized:
        if len(sent) != 0:
            v = sum([word_embeddings.get(w, np.zeros((50,))) for w in sent])/(len(sent)+0.001)
        else:
            v = np.zeros((50,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(summ_sent), len(summ_sent)])
    for i in range(len(summ_sent)):
        for j in range(len(summ_sent)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]

    # Converting sentences to a graph
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank_numpy(nx_graph)
    
    # Extracting top N sentences as summary
    summary = []
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(summ_sent)), reverse=True)
    for i in range(10):
        summary.append(ranked_sentences[i][1])

    return summary

def query_summary(text):
    query_tokens = tokenize(text)

    trump_summary = query_most_similar(query_token=query_tokens, 
                                       model=trump_d2v,
                                       corpus=trump_corpus, 
                                       data_lemmatized=trump_lemma)
    biden_summary = query_most_similar(query_token=query_tokens, 
                                       model=biden_d2v,
                                       corpus=biden_corpus, 
                                       data_lemmatized=biden_lemma)
    return({"Donald Trump": trump_summary, "Joe Biden": biden_summary})

    