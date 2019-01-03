from textCleaning import prepare_text_for_lda
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel


model= LdaModel.load("model20.gensim")

dictionary = corpora.Dictionary.load('dictionary.gensim')

# 'One of the misconceptions of many companies is that it is necessary to collect large amounts of data, 
# however, Tom Lawrie-Fussey, digital services specialist, Cambridge Design Partnership in the UK, warned 
# this is not always the best approach.'

new_doc = 'One of the misconceptions of many companies is that it is necessary to collect large amounts of data, however, Tom Lawrie-Fussey, digital services specialist, Cambridge Design Partnership in the UK, warned this is not always the best approach.'
print(new_doc)
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# print(model.get_document_topics(new_doc_bow))
for index, score in sorted(model[new_doc_bow], key=lambda tup: -1*tup[1]): #just model[new_doc_bow] if you dont wana sort
    print("Score: {}\n Topic: {}".format(score, model.print_topic(index, 5)))

# topics = model.print_topics(num_words=4)
# for topic in topics:
#     print(topic)



for idx, topic in model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")