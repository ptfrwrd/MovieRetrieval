from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import wordnet_ic
from pywsd.lesk import adapted_lesk

# расширенный Леск
def do_wsd(mass_words):
    res_synsets = []
    sentence = " ".join(mass_words)
    print(sentence)
    for word in mass_words:
        res_synsets.append(adapted_lesk(sentence, word))
    return res_synsets


def find_semantics(user_sentence, topic_sentence):
    mean_similarity = []
    for user_word in user_sentence:
        for topic_word in topic_sentence:
            similarity = adapted_lesk(topic_word, user_word)
            mean_similarity.append(similarity)
    return np.mean(mean_similarity)


if __name__ == '__main__':
    user_words = ["woman", "loner", "hotel", "escape", "find", "tell", "blind", "leave", "begin", "animal"]
    topic_words = ['girl']
    user_words_wsd = do_wsd(user_words)
    topic_words_wsd = do_wsd(topic_words)
    print(user_words_wsd)
    print(topic_words_wsd)
    #print(adapted_lesk(user_words[0], topic_words[0]))
    #sim = find_semantics(user_words_wsd, topic_words_wsd)
    #print(sim)
