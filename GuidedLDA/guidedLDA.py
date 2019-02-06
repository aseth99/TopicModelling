# # https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164
# import numpy as np
# import guidedlda
# from textCleaning import prepare_text_for_lda
# import gensim
# import pickle
# from gensim import corpora

# with open('corpus.pkl', 'rb') as pickle_file:
#     corpus = pickle.load(pickle_file)
# # corpus = pickle.load('corpus.pkl')
# dictionary = corpora.Dictionary.load('dictionary.gensim')
# # model= LdaModel.load("model5.gensim")

# # def bow_iterator(docs, dictionary):
# #     for doc in docs:
# #         yield dictionary.doc2bow(doc)

# # def get_term_matrix(msgs, dictionary):
# #     bow = bow_iterator(msgs, dictionary)
# #     X = np.transpose(matutils.corpus2csc(bow).astype(np.int64))
# #     return X

# # X = get_term_matrix(train_cleaned, dictionary)

# # model = guidedlda.GuidedLDA(alpha=.1, n_topics=NUM_TOPICS, n_iter=300, random_state=7, refresh=20)
# # model.fit(X, seed_topics=seed_topics, seed_confidence=0.6)


# new_doc = 'Topic models are built around the idea that the semantics of our document are actually being governed by some hidden, or “latent,” variables that we are not observing. As a result, the goal of topic modeling is to uncover these latent variables — topics — that shape the meaning of our document and corpus. The rest of this blog post will build up an understanding of how different topic models uncover these latent topics.'
# # print(new_doc)
# new_doc = prepare_text_for_lda(new_doc)
# new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
# # vocab = guidedlda.datasets.load_vocab('allArticles.txt')
# word2id = dict((v, idx) for idx, v in enumerate(dictionary))
# print(X[0])
# # print(new_doc_bow.shape)

# # print(new_doc_bow.sum())
# # Normal LDA without seeding
# model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
# model.fit(new_doc_bow)

# topic_word = model.topic_word_
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# # Guided LDA with seed topics.
# seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
#                    ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
#                    ['music', 'write', 'art', 'book', 'world', 'film'],
#                    ['political', 'government', 'leader', 'official', 'state', 'country', 'american', 'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

# model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

# seed_topics = {}
# for t_id, st in enumerate(seed_topic_list):
#     for word in st:
#         seed_topics[word2id[word]] = t_id

# model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

# n_top_words = 10
# topic_word = model.topic_word_
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
import numpy as np
import guidedlda

X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))

print(X.shape)

print(X.sum())
# Normal LDA without seeding
model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# Guided LDA with seed topics.
seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                   ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                   ['music', 'write', 'art', 'book', 'world', 'film'],
                   ['political', 'government', 'leader', 'official', 'state', 'country', 'american', 'case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))