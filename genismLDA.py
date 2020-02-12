from pprint import pprint
import os
# Import fix
try:
  from scipy.sparse.sparsetools.csr import _csr
except:
  from scipy.sparse import sparsetools as _csr
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import gensim, spacy
import re


def clean_data(description_movie):

    def sent_to_words(sentences):
            sentences = re.sub('\s+', ' ', sentences)
            sentences = re.sub("\'", "", sentences)
            sentences = gensim.utils.simple_preprocess(sentences, deacc=True)
            yield (sentences)


    data_words = list(sent_to_words(description_movie))
    return (data_words)



def process_words(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    def stop_words():
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'scene', ' film'])
        return stop_words
    stop_words = stop_words()

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]
    data_words = [bigram_mod[doc] for doc in data_words]
    data_words = [trigram_mod[bigram_mod[doc]] for doc in data_words]
    texts_out = []
    nlp = spacy.load(r'C:\Users\Ситилинк\AppData\Roaming\Python\Python38\site-packages\en_core_web_sm\en_core_web_sm-2.2.0', disable=['parser', 'ner'])
    for sent in data_words:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


def lda_model(data_ready):
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=1,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)
    pprint(lda_model.print_topics())
    return lda_model

def lda_model_mallet(data_ready):
    os.environ['MALLET_HOME'] = 'D:\mallet-2.0.8'
    mallet_path = r'D:\mallet-2.0.8\bin\mallet'
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=1, id2word=id2word)
    # Show Topics
    pprint(ldamallet.show_topics(formatted=False))
    return ldamallet



