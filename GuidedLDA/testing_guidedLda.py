import pickle
import numpy as np
from numpy import argsort
import guidedlda
import pickle
from gensim import corpora
from textCleaning import prepare_text_for_lda
from sklearn.feature_extraction.text import CountVectorizer

with open('guidedlda_model.pickle', 'rb') as file_handle:
    model = pickle.load(file_handle)

with open('sentence_data', 'rb') as pickle_file:
  sentence_data = pickle.load(pickle_file)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentence_data)
vocab = vectorizer.get_feature_names()

word2id = dict((v, idx) for idx, v in enumerate(vocab))

n_top_words = 5
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


doc_topic = model.transform(X)