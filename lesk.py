from nltk.corpus import wordnet as wn

def similarity_gloss(words):
    similarity = []
    for main_word in words:
        main_word = wn.synsets(main_word)[0]
        for second_word in words:
            second_word = wn.synsets(second_word)[0]
            similarity.append(main_word.lch_similarity(second_word))
    return similarity
