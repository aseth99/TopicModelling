import numpy as np
from numpy import argsort
import guidedlda
import pickle
from gensim import corpora
from gensim import matutils
from textCleaning import prepare_text_for_lda
from sklearn.feature_extraction.text import CountVectorizer

# X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
# vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
# word2id = dict((v, idx) for idx, v in enumerate(vocab))

with open('sentence_data', 'rb') as pickle_file:
  sentence_data = pickle.load(pickle_file)
# vocab2 = corpora.Dictionary.load('dictionary.gensim')
# # word2id = dict((v, idx) for idx, v in enumerate(vocab2))
# vocab = corpora.Dictionary.load_from_text('dictionary')
# print(word2id)
# print(corpus[:3])
# word_data = []
# text_data = []
# sentence_data = []
# counter = 0
# space = ' '
# with open('allArticles2.txt') as f:
#     for line in f:
#         # if random.random() > .99:
#         #  
#         print(counter)   
#         tokens = prepare_text_for_lda(line)
#         # print(tokens)
#         # for token in tokens:
#         #     word_data.append(token)
#         # text_data.append(tokens)
#         newTokens = space.join(tokens)
#         sentence_data.append(newTokens)
#         counter += 1


# dictionary = corpora.Dictionary(text_data)
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# print(len(word_data))
# unique_words = set(word_data)
# print(len(unique_words))
# vocabNew = []
# for x in dictionary.values():
#     vocabNew.append(x)
# print(vocabNew)
# print(len(vocabNew))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentence_data)
vocab = vectorizer.get_feature_names()
# print(vocabTest)
# print(len(vocabTest))
word2id = dict((v, idx) for idx, v in enumerate(vocab))


# dictionaryVocab = corpora.Dictionary(text_data)
# dictionaryVocab.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
# model= LdaModel.load("model5.gensim")

# def bow_iterator(docs, dictionary):
#     for doc in docs:
#         yield dictionary.doc2bow(doc)

# def get_term_matrix(msgs, dictionary):
#     bow = bow_iterator(msgs, dictionary)
#     X = np.transpose(matutils.corpus2csc(bow).astype(np.int64))
#     return X

# X = get_term_matrix(text_data, dictionaryVocab)

print(X.shape)

print(X.sum())
# Normal LDA without seeding
model = guidedlda.GuidedLDA(n_topics=10, n_iter=100, random_state=7, refresh=20)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    # print("printing i and topic_dist")
    # print(i)
    # print(topic_dist)
    # print(str(i))
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    # print(topic_words)
    # print(' '.join(topic_words))
    print('Topic {}: {}'.format(str(i), ' '.join(topic_words)))


# Guided LDA with seed topics.
# management changes
# business expansion
# competitive landscape
# Financial & Business Performance:
# health

seed_topic_list = [['manage', 'hire', 'depart', 'executive', 'director', 'ceo', 'internal'],
                   ['expand', 'factory', 'business', 'industry', 'hire'],
                   ['competitor', 'insights', 'share', 'product', 'launch', 'analyze', 'service', 'new'],
                   ['earning', 'outlook', 'sales', 'stock', 'dividends'],
                   ['health', 'study', 'sugar', 'sick',]]


counter = 0
for array in seed_topic_list:
    print("topic {}: {}".format(counter, array))
    counter += 1
model = guidedlda.GuidedLDA(n_topics=10, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.40)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


doc_topic = model.transform(X)
# for i in range(5):
#     print("top topic: {} Document: {}".format(doc_topic[i].argmax(), ', '.join(np.array(vocab)[list(reversed(X[i,:].argsort()))[0:5]])))
# print(doc_topic)
for i in range(10):
    print(doc_topic[i])
    print("top topic: {} words: ".format(doc_topic[i].argmax()))
    print(np.array(vocab)[np.argsort(topic_word[doc_topic[i].argmax()])][:-(n_top_words+1):-1])


with open('guidedlda_model10.pickle', 'wb') as file_handle:
    pickle.dump(model, file_handle)

