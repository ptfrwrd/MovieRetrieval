from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import wordnet_ic
from pywsd.lesk import adapted_lesk


# расширенный Леск
def do_wsd(mass_words):
    res_synsets = []
    sentence = " ".join(mass_words)
    for word in mass_words:
        res_synsets.append(adapted_lesk(sentence, word))
    return res_synsets


def find_semantics(user_sentence, topic_sentence):
    mean_similarity = []
    for user_w in user_sentence:
        temp_similarity = []
        for topic_w in topic_sentence:
            if user_w.name() == topic_w.name():
                temp_similarity.append(0)

            elif user_w.pos == topic_w.pos and (wn.synset(user_w.name()).wup_similarity(wn.synset(topic_w.name())) != None):
                similarity = wn.synset(user_w.name()).wup_similarity(wn.synset(topic_w.name()))
                temp_similarity.append(similarity)

            else:
                temp_similarity.append(1)
            
        print(temp_similarity)
        temp = min(temp_similarity)
        mean_similarity.append(temp)
    return np.mean(mean_similarity)


if __name__ == '__main__':
    user_words = ["love", "together", "good", "escape", "find", "tell", "blind", "leave", "begin", "animal"]
    topic_words = ["amour", "loner", "bad", "escape", "find", "tell", "blind", "leave", "begin", "animal"]
    user_words_wsd = do_wsd(user_words)
    topic_words_wsd = do_wsd(topic_words)
    print(user_words_wsd)
    print(topic_words_wsd)
    sim = find_semantics(user_words_wsd, topic_words_wsd)
    print(sim)
