import pandas as pd
import genismLDA as gn
import lesk as lk

def words(lda_model):
   topics = []
   for i in range(10):
      topics.append(lda_model.show_topics(formatted=False)[0][1][i][0])
   return topics


def weight_words(lda_model):
   weight = []
   for i in range(10):
      weight.append(lda_model.show_topics(formatted=False)[0][1][i][1])
   return weight


data = pd.read_csv('wiki_movie_plots_deduped.csv', encoding= 'unicode_escape')
description_movie = pd.DataFrame({"Plot": data['Plot'], "Title": data['Title']})
description_movie.dropna()
movies_count = len(description_movie)
for index in range(1):
   print(description_movie['Plot'][index])
   tokens = gn.clean_data(description_movie['Plot'][index])
   data_ready = gn.process_words(tokens)
   lda_model = gn.lda_model(data_ready)
   topics = words(lda_model)
   weight = weight_words(lda_model)

   #print(lda_model.get_topic_terms(0))
   #print(lda_model.show_topics(formatted=False)[0][1][1][0])
   #work
   #lda_mallet = gn.lda_model_mallet(data_ready)


