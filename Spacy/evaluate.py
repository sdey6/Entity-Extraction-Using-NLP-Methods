import ast
import logging
import re
import string

import numpy as np
import pandas as pd
from spacy.scorer import Scorer
from spacy.training import Example

from named_entity_extraction.Config.config import pickle_spacy_path, test_data_path
from named_entity_extraction.Spacy.Model_Training.prepare_train_data import df_spacy_pp
from named_entity_extraction.Spacy.predict_evaluate.predict import load_model
from sklearn.model_selection import train_test_split
# evaluate performance on training data

model = load_model(pickle_spacy_path)


def get_eval_format_train(df):
    """prepare data for evaluating of trained model"""
    eval_list = []
    for i, j in df.iterrows():
        title = j['title']
        spacy = ast.literal_eval(j['spacy'])
        tup1 = (title, spacy)
        eval_list.append(tup1)
    return eval_list


def get_score(df):
    """get training score on Metrics"""
    eval_list = get_eval_format_train(df)
    examples_l = []
    scorer = Scorer()
    for input_, annot in eval_list:
        doc_gold_text = model.make_doc(input_)
        example = Example.from_dict(doc_gold_text, {"entities": annot})
        example.predicted = model(str(example.predicted))
        examples_l.append(example)
    logging.getLogger("spacy_lefff").setLevel(logging.WARNING)
    return scorer.score(examples_l)


# evaluate on test data

def preprocessing_test_data(text):
    """to preprocess the test data"""
    text = str(text)
    text = text.lower().replace('[{}]'.format(string.punctuation), '')
    text = re.sub(r'[^\w\s]', '', text)
    return text


def apply_preprocess():
    df_test = pd.read_csv(test_data_path, index_col=0)
    df_test['Title'] = df_test['Title'].apply(preprocessing_test_data)
    df_test['Innovation_gold'] = df_test['Innovation_gold'].apply(preprocessing_test_data)
    return df_test


def predict_on_test(df):
    """predict on test data by trained spacy model with hand annotated data"""
    list_pred_gold = []
    for i, j in df.iterrows():
        inp = j['Title']
        doc = model(inp)
        for ent in doc.sents:
            list_pred_gold.append({'Title': j['Title'], 'Innovation_gold': j['Innovation_gold'],
                                   'Innovation_predicted': (
                                       ' '.join(str(v) for v in ent.ents), *(e.label_ for e in ent.ents))})
    df = pd.DataFrame(list_pred_gold)
    return df


def predict_on_test_gpt():
    """predict on test data by trained spacy model with gpt annotated data"""
    path_gpt = 'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction/Evaluation/gpt_output_processed.csv'
    df_gpt_output = pd.read_csv(path_gpt)
    list_pred_gold = []
    for i, j in df_gpt_output.iterrows():
        inp = j['Title']
        doc = model(inp)
        for ent in doc.sents:
            list_pred_gold.append({'Title': j['Title'], 'Innovation_gold': j['Innovation'],
                                   'Innovation_predicted': (
                                       ' '.join(str(v) for v in ent.ents), *(e.label_ for e in ent.ents))})
    df = pd.DataFrame(list_pred_gold)
    return df


def make_eval_data_spacy(df):
    """prepare data format for evaluation"""
    pred_list = []
    inno_list = []
    example_list = []
    t_1 = ()
    t_2 = ()
    for i, j in df.iterrows():
        out_i = j['Innovation_predicted'][0]
        out_e = j['Innovation_predicted'][-1]
        l_1 = list(t_1)
        l_1.insert(0, out_e)
        l_1.insert(1, out_i)
        new1 = tuple(l_1)
        pred_list.append([new1])
        l_2 = list(t_2)
        l_2.insert(0, 'INNOVATION')
        l_2.insert(1, j['Innovation_gold'])
        new2 = tuple(l_2)
        inno_list.append([new2])
        example_list.append(j['Title'])
    return inno_list, pred_list, example_list


def flat_accuracy(df):
    test_acc_list = []
    for i, j in df.iterrows():
        title = j['Title']
        annot = j['Innovation_gold']
        prediction = model(title)
        for ent in prediction.ents:
            if annot == ent.text:
                test_acc_list.append(1)
            else:
                test_acc_list.append(0)
    return test_acc_list


accuracy = round(sum(flat_accuracy(apply_preprocess())) / len(apply_preprocess()) * 100, 2)

accuracy_list = []
for i, j in apply_preprocess().iterrows():
    title = j['Title'].strip()
    annot = j['Innovation_gold']
    prediction = model(title)
    for ent in prediction.ents:
        if title.find(ent.text) != -1:
            len_gold = len(annot.split())
            len_pred = len((ent.text).split())
            acc_per_example = len_pred / len_gold
            if acc_per_example < 1:
                accuracy_list.append(acc_per_example)
mean_acc = np.mean(accuracy_list).round(2)



