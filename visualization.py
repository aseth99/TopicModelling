import pyLDAvis.gensim
import gensim 
import pickle

# pyLDAvis.enable_notebook()
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
# lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# lda_html = pyLDAvis.prepared_data_to_html(lda_display)
# pyLDAvis.show(lda_html)
# pyLDAvis.display(lda_display)
# pyLDAvis.save_html(lda_html, 'ldaHTML.html')

print('\nPerplexity: ', lda.log_perplexity(corpus)) 