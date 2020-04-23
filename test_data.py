import pandas as pd
import random

def delete_sent(plot):
    sentense_num = random.randint(0, len(plot) - 1)
    del (plot[sentense_num])
    return plot

if __name__ == '__main__':
    data = pd.read_csv('wiki_movie_plots_deduped.csv', encoding='unicode_escape')
    data = data.loc[data['Release Year'] == 2016]
    description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
    description_movie = description_movie.dropna()
    description_movie = description_movie[description_movie['Plot'].apply(len) > 400]
    description_movie = description_movie.reset_index()

    for index in range(len(description_movie)):
        plot = description_movie['Plot'][index]
        plot = plot.split('.')
        sentense_num = random.randint(0, len(plot)-1)
        del(plot[sentense_num])
        sentense_num = random.randint(0, len(plot)-1)
        del (plot[sentense_num])
        sentense_num = random.randint(0, len(plot) - 1)
        del (plot[sentense_num])

        plot = ".".join(plot)
        description_movie['Plot'][index] = plot
    description_movie.to_csv("test_2016.csv", encoding='unicode_escape')

