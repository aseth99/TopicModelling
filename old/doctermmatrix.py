# import shorttext
# import gensim
# import pickle
# from gensim import corpora

# with open('corpus.pkl', 'rb') as pickle_file:
#     corpus = pickle.load(pickle_file)
# # corpus = pickle.load('corpus.pkl')
# dictionary = corpora.Dictionary.load('dictionary.gensim')
# docids = []

# for x in range(len(corpus)):
# 	docids.append(x)
# print(corpus[0])
# usprez_dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids)

import textmining

save_path = '/Users/aman/Desktop/Hammer/TopicModelling/Articles'
completeName1 = os.path.join(save_path,'1.txt')
completeName2 = os.path.join(save_path,'2.txt')
completeName3 = os.path.join(save_path,'3.txt')


def termdocumentmatrix_example():
    # Create some very short sample documents
    # doc1 = 'John and Bob are brothers.'
    # doc2 = 'John went to the store. The store was closed.'
    # doc3 = 'Bob went to the store too.'
    # Initialize class to create term-document matrix
    tdm = textmining.TermDocumentMatrix()
# Add the documents
    tdm.add_doc(completeName1)
    tdm.add_doc(completeName2)
    tdm.add_doc(completeName3)

    # tdm.add_doc(doc2)
    # tdm.add_doc(doc3)
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('matrix.csv', cutoff=1)
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
    for row in tdm.rows(cutoff=1):
        print(row)