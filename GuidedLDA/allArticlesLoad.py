# basically main function, prints out every 100 lines to visualize tokens, text_data is a list of lists of tokens
from textCleaning import prepare_text_for_lda
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pickle
import random

text_data = []
# sentence_data = []
counter = 0
space = ' '
with open('allArticles.txt') as f:
    for line in f:
        # if random.random() > .99:
        #     
        print(counter)   
        tokens = prepare_text_for_lda(line)
        # print(tokens)
        
        text_data.append(tokens)
        # newTokens = space.join(tokens)
        # sentence_data.append(newTokens)
        counter += 1


dictionary = corpora.Dictionary(text_data)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
# corpus = [dictionary.doc2bow(text) for text in text_data]
# pickle.dump(corpus, open('corpus.pkl', 'wb'))
# dictionary.save('dictionary.gensim')
dictionary.save_as_text('dictionary')
