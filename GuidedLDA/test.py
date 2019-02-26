from textCleaning import prepare_text_for_lda
import gensim
from gensim import corpora
import pickle

sentence_data = []
counter = 0
space = ' '
with open('allArticles2.txt') as f:
    for line in f:
        # if random.random() > .99:
        #  
        print(counter)   
        tokens = prepare_text_for_lda(line)
        # print(tokens)
        # for token in tokens:
        #     word_data.append(token)
        # text_data.append(tokens)
        newTokens = space.join(tokens)
        sentence_data.append(newTokens)
        counter += 1

with open('sentence_data2', 'wb') as fp:
    pickle.dump(sentence_data, fp)

#guidedLDA try with tfidf as keywords
#webscrapers
