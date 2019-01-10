import pyLDAvis.gensim
import gensim 
import pickle
from gensim.models import CoherenceModel

# pyLDAvis.enable_notebook()
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
# lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# lda_html = pyLDAvis.prepared_data_to_html(lda_display)
# pyLDAvis.show(lda_html)
# pyLDAvis.display(lda_display)
# pyLDAvis.save_html(lda_html, 'ldaHTML.html')

#coherence score returns "nan" ...?
# coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=Dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)

#this works, not sure what perplexity determines... how accurate it is but not sure how its calculated/measured
# print('\nPerplexity: ', lda.log_perplexity(corpus)) 

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(lda_display, 'ldaHTML.html')
