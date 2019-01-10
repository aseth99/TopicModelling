import gensim
import pickle
from gensim import corpora k

with open('corpus.pkl', 'rb') as pickle_file:
    corpus = pickle.load(pickle_file)
# corpus = pickle.load('corpus.pkl')
dictionary = corpora.Dictionary.load('dictionary.gensim')

bow_doc_4310 = corpus[10]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], dictionary[bow_doc_4310[i][0]], bow_doc_4310[i][1]))
# tfidf = models.TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
# from pprint import pprint
# for doc in corpus_tfidf:
#     pprint(doc)
#     break