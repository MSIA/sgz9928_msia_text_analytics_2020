# import modules & set up logging
import os
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


def word2vec_model(sentences, min_count, workers, window, size, sg):
    # Training word2vec model
    model = gensim.models.Word2Vec(sentences=sentences, min_count=min_count, workers=workers, window=window, size=size, sg=sg)
    if sg==1:
        model_file = 'trained_models/skipgram_mincount'+str(min_count)+'_size'+str(size)+'_window'+str(window)
    else:
        model_file = 'trained_models/cbow_mincount'+str(min_count)+'_size'+str(size)+'_window'+str(window)
    model.save(model_file)

    # Testing Embeddings
    report_file = open(model_file+'.txt', "w")
    print(f'Top 10 most similar words to "gun":\n {model.most_similar(["gun"], topn=20)}', file=report_file)
    print(f'\nTop 10 most similar words to "america":\n {model.most_similar(["america"], topn=20)}', file=report_file)
    print(f'\nTop 10 most similar words to "peace":\n {model.most_similar(["peace"], topn=20)}', file=report_file)
    print(f'\nTop 10 most similar words to "safety":\n {model.most_similar(["safety"], topn=20)}', file=report_file)
    print(f'\nTop 10 most similar words to "violence":\n {model.most_similar(["violence"], topn=20)}', file=report_file)


if __name__ == "__main__":
    sentences = MySentences('processed_files') # a memory-friendly iterator
    # Skip gram model
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=2, 
                   size=50, 
                   sg=1)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=2, 
                   size=100, 
                   sg=1)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=3, 
                   size=150, 
                   sg=1)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=5, 
                   size=150, 
                   sg=1)

    # CBOW model
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=2, 
                   size=50, 
                   sg=0)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=2, 
                   size=100, 
                   sg=0)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=3, 
                   size=150, 
                   sg=0)
    word2vec_model(sentences=sentences, 
                   min_count=10, 
                   workers=8, 
                   window=5, 
                   size=150, 
                   sg=0)





