from multiprocessing.dummy import freeze_support
import pandas as pd
import alg_lesk as lesk
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
import matplotlib.pyplot as plt
import numpy as np
import random
from nltk.corpus import wordnet as wn



def clean_data(text):
    def sentences_to_words(sentences):
        # delete new lines
            sentences = re.sub('\s+', ' ', sentences)
        # delete ", '
            sentences = re.sub("\'", "", sentences)
        # delete proper nouns
            sentences = re.sub(r'([^.])( [A-Z]\w*)', r'\1', sentences)
        # tokens without . ,
            sentences = gensim.utils.simple_preprocess(sentences, deacc=True)
            yield (sentences)
    clear_data = list(sentences_to_words(text))
    return (clear_data)


# create N-gramms
def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    bigrams_text = [bigram_mod[doc] for doc in texts]
    trigrams_text =  [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
    return trigrams_text


# final processing, includes lemmatization and stop-words
def process_words(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'scene', ' film', 'irish',
                       'eytinge', 'soon', 'several', 'priscilla', 'pink', 'orange', 'blonde'])
    text_without_stop_words = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in text]
    texts_bigrams = make_bigrams(text_without_stop_words)
    nlp = spacy.load(r'C:\Users\Ситилинк\AppData\Roaming\Python\Python38\site-packages\en_core_web_sm\en_core_web_sm-2.2.0',
                     disable=['parser', 'ner'])
    data_lemmatized = []
    for sent in texts_bigrams:
        doc = nlp(" ".join(sent))
        data_lemmatized.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    data_lemmatized = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_lemmatized]
    return data_lemmatized


# create lda model and score
def create_lda_model(data_ready):
    id2word = corpora.Dictionary(data_ready)
    corpus = [id2word.doc2bow(text) for text in data_ready]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                                    update_every=1, chunksize=1, iterations=100)
    # a measure of how good the model is. lower the better.
    #print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    #coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_ready,  coherence='c_v')
    #coherence_lda = coherence_model_lda.get_coherence()

    #print('\nCoherence Score: ', coherence_lda)
    #print('\n Topics: ', lda_model.print_topics())

    return lda_model #, coherence_lda


def show_topics(lda_model_res):
   topics = []
   for i in range(10):
      topics.append(lda_model_res.show_topics(formatted=False)[0][1][i][0])
   return topics


def remove_char(s):
    result = s[1: -1]
    return result


def create_topic(topic):
    topic = remove_char(topic)
    topic = topic.split(", ")
    new_topic = []
    for el in topic:
        el = remove_char(el)
        new_topic.append(el)
    return new_topic


if __name__ == '__main__':
    lda_data = pd.read_csv('LDA.csv', encoding='unicode_escape')
    test_data = pd.read_csv('test_half_4.csv', encoding='unicode_escape')
    precision, TP, FP = 0, 0, 0
    for j in range(len(test_data) // 3):
        print("итарация: ", j)
        user_description = test_data['Plot'][j]
        expected_response = test_data['Title'][j]
        user_description = clean_data(user_description)
        user_description = process_words(user_description)
        lda_user = create_lda_model(user_description)
        user_topics = show_topics(lda_user)
        lesk_user_topics = lesk.do_wsd(user_topics)
        lesk_user_topics = list(filter(None, lesk_user_topics))
        similarity = []
        for i in range(len(lda_data)):
            topic_lda = lda_data['Topic_LDA'][i]
            topic_lda = create_topic(topic_lda)
            lesk_topics = lesk.do_wsd(topic_lda)
            lesk_topics = list(filter(None, lesk_topics))
            similarity_temp = lesk.find_semantics(lesk_user_topics, lesk_topics)
            similarity.append(similarity_temp)
        film_index = similarity.index(min(similarity))
        algorithm_response = test_data['Title'][film_index]
        similarity_response = min(similarity)
        if expected_response == algorithm_response:
            TP += 1
        else:
            FP += 1
    precision = TP/(TP + FP)
    print(precision)


