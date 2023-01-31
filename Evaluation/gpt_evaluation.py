import re
import time
from getpass import getpass
import openai
import pandas as pd
import pickle
import json

from eval4ner import muc

path = 'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction/sample_files/test_set_100.csv'
df = pd.read_csv(path, index_col=0)
df_inno = df.iloc[:, :2]
print(f'Shape of the test dataframe with innovation entity: {df_inno.shape}')

my_key = ''
with open(
        'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction/Config/JSON/gpt_api_key.json',
        'r') as api_file:
    json_data = json.load(api_file)
    my_key = json_data["API_KEY"]


def few_shot_gpt(df):
    openai.api_key = my_key
    list_innov = []
    for i, j in df.iterrows():
        text = j['Title']
        response = openai.Completion.create(
            model="text-curie-001",
            prompt="""
              [Text]: "Ghost particles" detected in the Large Hadron Collider for first time
              [Innovation]: Ghost particles
              [Material] :none
              [utilization]: none
              ###
              [Text]: New explanation for Antarctic mystery doesn't need "parallel universe"
              [Innovation]: none
              [Material] :none
              [utilization]: none
              ###
              [Text]: Atomic tractor beam traps atoms for quantum memory
              [Innovation]: Atomic tractor beam
              [Material] :none
              [utilization]: none          
              ###
              [Text]: Ricoh to preview GRIII compact camera at Photokina 2018
              [Innovation]: GRIII compact camera
              [Material] :none
              [utilization]: none
              ###
              [Text]:Fifth-dimensional black hole could cause general relativity to break down
              [Innovation]: none
              [Material] :none
              [utilization]: none
              ###
              [Text]: MIT and Harvard study unpacks the push and pull of diet and exercise
              [Innovation]:none
              [Material] :none
              [utilization]: none
              ###
              [Text]: "Vegan spider silk" offers a plant-based replacement for common plastic
              [Innovation]:Vegan spider silk
              [Material] :plant-based 
              [utilization]: plastic
              ###
              [Text]: Can crowdfunding give us safe fusion power by 2020?
              [Innovation]: none
              [Material] :none
              [utilization]:  none
              ###
              [Text] : Fronius rolls out its first customer SolHub solar-to-hydrogen station
              [Innovation]: solar-to-hydrogen station
              [Material] :none
              [utilization]: none
              ###
              [Text]:Cancer's genetic secrets revealed through massive international study
              [Innovation]: none
             [Material] :none
              [utilization]: none
              ###
              [Text]: Simple EEG brain scan can tell if antidepressant drugs work for you
              [Innovation]: none
              [Material] :none
              [utilization]: none
              ###
              [Text]: Uncanny visions in the 2022 Urban Photo Awards
              [Innovation]: none
              [Material] :none
              [utilization]: none
              ###
              [Text]:%s""" % text,
            temperature=0.3,
            max_tokens=150,
            top_p=0.5,
            frequency_penalty=0.0,
            presence_penalty=0.0)
        list_innov.append(response["choices"][0]["text"])
        time.sleep(3)
    return list_innov


#res = few_shot_gpt(df_inno)
#pickle.dump(res, open('./test_gpt_output.pickle', 'wb'))
op = pickle.load(open(
    'C:/Users/souri/PycharmProjects/Entity-Extraction-Using-NLP-Methods/named_entity_extraction/Evaluation/test_gpt_output.pickle',
    'rb'))
df_gpt = pd.DataFrame({'Text': df_inno['Title'], 'Innovation_predicted': op})


def post_process_gpt(df):
    df_temp = df["Innovation_predicted"].str.split("\n", expand=True)
    df['Innovation'] = df_temp[1]
    df['Material'] = df_temp[2]
    df['Utilization'] = df_temp[3]
    df.drop('Innovation_predicted', axis=1, inplace=True)
    df.replace("\[[\w]+\] ?:", "", regex=True, inplace=True)
    return df


gpt_output_data = post_process_gpt(df_gpt)
print(gpt_output_data.head())
gpt_output_data.to_csv('./gpt_output.csv')


def post_process(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub("\[[\w]+\] ?:", "", text)
    text = text.strip()
    return text


gpt_output_data['Innovation'] = gpt_output_data['Innovation'].apply(lambda x: post_process(x))
df_inno['Title'] = df_inno['Title'].apply(lambda x: post_process(x))
df_inno['Innovation_gold'] = df_inno['Innovation_gold'].apply(lambda x: post_process(x))

df_inno = df_inno.join(gpt_output_data['Innovation'])


def post_process_2(text):
    text = text.replace(r'console', '')
    text = text.strip()
    return text


df_inno['Innovation'] = df_inno['Innovation'].apply(lambda x: post_process_2(x))

df_inno.to_csv('./gpt_output_processed.csv')


def make_eval_data(df):
    gold_inno_list = []
    pred_inno_list = []
    text_list = []
    for i, j in df.iterrows():
        gold = j['Innovation_gold']
        tup_1 = ()
        tup_2 = ()
        l1 = list(tup_1)
        l1.insert(0, 'INNOVATION')
        l1.insert(1, gold)
        l1_tup = tuple(l1)
        gold_inno_list.append([l1_tup])
        l2 = list(tup_2)
        l2.insert(0, 'INNOVATION')
        l2.insert(1, j['Innovation'])
        l2_tup = tuple(l2)
        pred_inno_list.append([l2_tup])
        text_list.append(j['Title'])
    return text_list, gold_inno_list, pred_inno_list


text_list, gold_inno_list, pred_inno_list = make_eval_data(df_inno)
print(f'The golden thruth list for evaluation is: {gold_inno_list}')
print(f'The predicted list for evaluation is: {pred_inno_list}')
print(f'The Examples for evaluation are:{text_list}')
print(len(pred_inno_list), len(gold_inno_list), len(text_list))


# The score on MUC metric for GPT as a model

def get_predictive_score(pred, actual, example):
    """get the MUC score from test data"""
    score = muc.evaluate_all(pred, actual * 1, example, verbose=False)
    return score


score_gpt = get_predictive_score(pred_inno_list, gold_inno_list, text_list)
print(f'The MUC score for GPT is:{score_gpt}')
