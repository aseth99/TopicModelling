from textCleaning import prepare_text_for_lda
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel


model= LdaModel.load("model20.gensim")

dictionary = corpora.Dictionary.load('dictionary.gensim')

# 'One of the misconceptions of many companies is that it is necessary to collect large amounts of data, 
# however, Tom Lawrie-Fussey, digital services specialist, Cambridge Design Partnership in the UK, warned 
# this is not always the best approach.'

# new_doc = 'Topic models are built around the idea that the semantics of our document are actually being governed by some hidden, or “latent,” variables that we are not observing. As a result, the goal of topic modeling is to uncover these latent variables — topics — that shape the meaning of our document and corpus. The rest of this blog post will build up an understanding of how different topic models uncover these latent topics.'
new_doc = input("\nWhat is the phrase of which topic you want? (type 'default' if you want preset paragraph) \n\n\n")
if new_doc == "default":
	new_doc = 'Topic models are built around the idea that the semantics of our document are actually being governed by some hidden, or “latent,” variables that we are not observing. As a result, the goal of topic modeling is to uncover these latent variables — topics — that shape the meaning of our document and corpus. The rest of this blog post will build up an understanding of how different topic models uncover these latent topics.'
	print(new_doc)
# print(new_doc)
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# print(model.get_document_topics(new_doc_bow))
for index, score in sorted(model[new_doc_bow], key=lambda tup: -1*tup[1]): #just model[new_doc_bow] if you dont wana sort
    print("Score: {}\n Topic: {}".format(score, model.print_topic(index, 5)))
    print("\n\n")

# topics = model.print_topics(num_words=4)
# for topic in topics:
#     print(topic)



# for idx, topic in model.print_topics(-1):
#     print("Topic: {} \nWords: {}".format(idx, topic ))
#     print("\n")