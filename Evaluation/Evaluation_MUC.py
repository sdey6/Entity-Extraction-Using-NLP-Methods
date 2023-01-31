import pickle

from eval4ner import muc
from transformers import BertTokenizerFast
import spacy
from torch import cuda
import torch
from spacy.training import biluo_tags_to_offsets
import io
from named_entity_extraction.BERT.predict import predict_sent
from named_entity_extraction.Spacy.predict_evaluate.evaluate import apply_preprocess, predict_on_test_gpt
from named_entity_extraction.Spacy.predict_evaluate.evaluate import predict_on_test, make_eval_data_spacy


def get_predictive_score(pred, actual, example):
    """get the MUC score from test data"""
    score = muc.evaluate_all(pred, actual * 1, example, verbose=False)
    return score


# Formatting for Evaluation spacy on hand annotated data

df_pred_spacy = predict_on_test(apply_preprocess())
df_pred_gold_spcy_gpt = predict_on_test_gpt()
inno_list, pred_list, example_list = make_eval_data_spacy(df_pred_spacy)
inno_list_2, pred_list_2, example_list_2 = make_eval_data_spacy(df_pred_gold_spcy_gpt)

print(f'Example test sentence:{example_list}')
print(f'Gold annotation:{inno_list}')
print(f'predicted annotation by Spacy:{pred_list}')
print(f'length of the lists created:{len(example_list)}, {len(inno_list)}, {len(pred_list)}')

print(f'Example test sentence:{example_list_2}')
print(f'Gold annotation:{inno_list_2}')
print(f'predicted annotation by Spacy:{pred_list_2}')
print(f'length of the lists created:{len(example_list_2)}, {len(inno_list_2)}, {len(pred_list_2)}')

# Get MUC score Spacy
score_spacy = get_predictive_score(pred_list,inno_list,example_list)
print(f'The MUC score for spacy on hand annotated data is:{score_spacy}')

score_spacy_throug_gpt = get_predictive_score(pred_list_2,inno_list_2,example_list_2)
print(f'The MUC score for spacy on gpt annotated data is:{score_spacy_throug_gpt}')



