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
import matplotlib.pyplot as plt
import numpy as np


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
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, texts=data_ready,  coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)
    print('\n Topics: ', lda_model.print_topics())

    return lda_model, coherence_lda


if __name__ == '__main__':

    data = pd.read_csv('wiki_movie_plots_deduped.csv', encoding='unicode_escape')
    data = data.loc[data['Release Year'] == 2015]
    description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
    description_movie = description_movie.dropna()
    description_movie = description_movie[description_movie['Plot'].apply(len) > 400]
    description_movie = description_movie.reset_index()
    max_index = []
    '''for index in range(len(description_movie)):
        plot = description_movie['Plot'][index]
        plot = clean_data(plot)
        plot = process_words(plot)
        lda = create_lda_model(plot)
        print("Итерация = ", index, "  индекс = ", lda[1])
        max_index.append(lda[1] + 1)

    res_index = np.mean(max_index)
    print("средний индекс: ", res_index)
'''
    user_description = input()
    user_description = clean_data(user_description)
    user_description = process_words(user_description)
    lda_res = create_lda_model(user_description)
