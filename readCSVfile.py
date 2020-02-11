import pandas as pd
import genismLDA as gn

data = pd.read_csv('wiki_movie_plots_deduped.csv')
description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
description_movie.dropna()
movies_count = len(description_movie)

for index in range(1):
   print(description_movie['Plot'][index])
   tokens = gn.clean_data(description_movie['Plot'][index])
   data_ready = gn.process_words(tokens)
   lda_mod = gn.lda_model(data_ready)





