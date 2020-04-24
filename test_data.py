import pandas as pd
import random

def delete_sent(plot):
    sentense_num = random.randint(0, len(plot) - 1)
    del (plot[sentense_num])
    return plot

if __name__ == '__main__':
    data = pd.read_csv('wiki_movie_plots_deduped.csv', encoding='unicode_escape')
    data = data.loc[data['Release Year'] == 2015]
    description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
    description_movie = description_movie.dropna()
    description_movie = description_movie[description_movie['Plot'].apply(len) > 400]
    description_movie = description_movie.reset_index()
    for index in range(len(description_movie)):
        plot = description_movie['Plot'][index]
        plot = plot.split('.')
        half_plot = len(plot) // 4
        if half_plot >= 2:
            start_i = random.randint(0, half_plot - 1)
            for i in range(int(start_i), half_plot):
                plot.remove(plot[i])

        plot = ".".join(plot)
        description_movie['Plot'][index] = plot
    description_movie.to_csv("test_4_rand.csv", encoding='unicode_escape')

