#unsupervised learning.. starting from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

import spacy
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English


def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

#use nltks wordnet to find synonyms, antonyms, basic meaning of words
# nltk.download('wordnet')
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
#WordNetLemmatizer gets root word
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

#filters out stopwords
# nltk.download('stopwords')


#prepare text for topic modelling
def prepare_text_for_lda(text):
    en_stop = set(nltk.corpus.stopwords.words('english'))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens