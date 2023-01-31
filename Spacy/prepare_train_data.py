import pandas as pd
import re
import ast

path_mapegy = 'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction\Spacy' \
              '/Model_Training/data/few_shot_post_processed.csv'
path_news = 'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction\Spacy' \
            '/Model_Training/data/new_Atlas_postprocessed.csv'
df_mapegy = pd.read_csv(path_mapegy)
df_news = pd.read_csv(path_news)
print(f'shape of mapegy data:{df_mapegy.shape} and news data:{df_news.shape}')
df_total = pd.concat([df_mapegy, df_news], ignore_index=True)
print(f'shape of combined data:{df_total.shape}')
path_final = 'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction\Spacy' \
             '/Model_Training/data/final_post_processed.csv'
df_spacy_pp = pd.read_csv(path_final, index_col=0)


def check(df):
    """Prepares data for spacy training"""
    final_list = []
    for i, j in df.iterrows():
        title = j['title']
        annot = ast.literal_eval(j['annotation'])[0][0]
        if title.find(annot) != -1:
            start_index = title.find(annot)
            end_index = title.find(annot) + len(annot)
            # print(f'{title}::{annot}')
            innov_entity = (start_index, end_index, 'INNOVATION')
            train_data = (title, {'entities': [innov_entity]})
            final_list.append(train_data)
    return final_list


data_training = check(df_spacy_pp)

