"""
    similarityextLesk(t1, t2) = overlap(gloss(t1), gloss(t2)+
    overlap(gloss(hyppo(t1)), gloss(hyypp(t2))+
    overlap(gloss(hyppo(t1)), gloss(t2))+
    overlap(gloss(t1), gloss(hyppo(t2)))

    overlap(t1, t2) — количество совпадений между термами t1 и t2,
    gloss(t) — определение терма t,
    hyppo(t) — гипероним слова, например для слова «красный» гиперонимом является слово «цвет»,
    t1 и t2 — термины.
"""

from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import wordnet_ic


def similarity_gloss(user_words, topic_words):
    similarity = []
    for main_word in user_words:
        main_word = wn.synsets(main_word)
        for synsets_mw in main_word:
            for second_word in topic_words:
                    second_word = wn.synsets(second_word)
                    temp_similarity = []
                    for synsets_sw in second_word:
                        if  synsets_mw == synsets_sw:
                            temp_similarity.append(0)
                            break
                        else:
                            if synsets_mw._pos == synsets_sw._pos and (synsets_mw.wup_similarity(synsets_sw) != None and synsets_mw.path_similarity(synsets_sw) != None):
                                            temp_similarity.append(synsets_mw.wup_similarity(synsets_sw) + synsets_mw.path_similarity(synsets_sw))
                            else:
                                temp_similarity.append(2)
            if temp_similarity:
                similarity.append(min(temp_similarity))
    return np.mean(similarity)


if __name__ == '__main__':
    user_words = ['loiter']
    topic_words = ['colour']

    sim = similarity_gloss(user_words, topic_words)
    print(sim)
