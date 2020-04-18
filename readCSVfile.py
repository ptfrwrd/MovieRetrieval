from multiprocessing.dummy import freeze_support
import pandas as pd
import lesk as lk
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
import numpy as np


def clean_data(description_movie):

    def sent_to_words(sentences):

            sentences = re.sub('\s+', ' ', sentences)
            sentences = re.sub("\'", "", sentences)
            sentences = re.sub(r'([^.])( [A-Z]\w*)', r'\1', sentences)
            sentences = gensim.utils.simple_preprocess(sentences, deacc=True)
            yield (sentences)
    data_words = list(sent_to_words(description_movie))
    return (data_words)


def process_words(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    def stop_words():
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'scene', ' film', 'irish',
                           'alice', 'rockies', 'eytinge', 'soon', 'several',  'priscilla'])
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
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                iterations=100,
                                                )
    #pprint(lda_model.print_topics())
    # Compute Perplexity
    #print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    #coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_ready,  coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()
    #print('\nCoherence Score: ', coherence_lda)
    return lda_model


def words(lda_model_res):
   topics = []
   for i in range(10):
      topics.append(lda_model_res.show_topics(formatted=False)[0][1][i][0])
   return topics

"""
def weight_words(lda_model_res):
   weight = []
   for i in range(10):
      weight.append(lda_model_res.show_topics(formatted=False)[0][1][i][1])
   return weight
"""

def write_lda_rezults(description_movie):
    tokens = clean_data(description_movie['Plot'][index])
    data_ready = process_words(tokens)
    lda_model_res = lda_model(data_ready)
    topics = words(lda_model_res)
    return topics


if __name__ == '__main__':
    freeze_support()
    data = pd.read_csv('wiki_movie_plots_deduped.csv', encoding= 'unicode_escape')
    description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
    description_movie = description_movie.dropna()
    description_movie = description_movie[description_movie['Plot'].apply(len) > 400]
    description_movie.index = range(len(description_movie))
    print(description_movie['Plot'][204])
    user_description = input()
    tokens = clean_data(user_description)
    print(tokens)
    data_ready = process_words(tokens)
    print(data_ready)
    description_res = lda_model(data_ready)
    user_topics = words(description_res)
    print(user_topics)
    movies_count = len(description_movie)
    recommendations = []
    films_names = []
    for index in range(3140, movies_count):
        base_topics = write_lda_rezults(description_movie)
        similarity = lk.similarity_gloss(user_topics, base_topics)
        recommendations.append(similarity)
        films_names.append(description_movie['Title'][index])
        print(index)
    print(films_names[recommendations.index(min(recommendations))])








