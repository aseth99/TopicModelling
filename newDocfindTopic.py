from textCleaning import prepare_text_for_lda
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel


model= LdaModel.load("model5.gensim")

dictionary = corpora.Dictionary.load('dictionary.gensim')

new_doc = 'One of the misconceptions of many companies is that it is necessary to collect large amounts of data, however, Tom Lawrie-Fussey, digital services specialist, Cambridge Design Partnership in the UK, warned this is not always the best approach.'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(model.get_document_topics(new_doc_bow))

topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic)