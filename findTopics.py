import gensim
import pickle
from gensim import corpora

with open('corpus.pkl', 'rb') as pickle_file:
    corpus = pickle.load(pickle_file)
# corpus = pickle.load('corpus.pkl')
dictionary = corpora.Dictionary.load('dictionary.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)