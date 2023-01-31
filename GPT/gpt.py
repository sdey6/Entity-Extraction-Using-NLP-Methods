
import pandas as pd
import string
import re
import nltk
import time
import openai

# client = nlpcloud.Client("gpt-j", "4af817d8d2338b9b7973974355d819dd62ac1dfe", gpu=True)
#
#
# def few_shot_nlp_cloud(df_demo):
#     list_innov = []
#     for i, j in df_demo.iterrows():
#         text = j['title']
#         # print('The example is:', text)
#         generation_tech = client.generation("""
#         [Text]: Taiwanese iPhone Supplier Foxconn Announces Entry into EV Market, Shows 3 Concept Vehicles
#           [Innovation]: EV, Concept Vehicles
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: New algorithm identifies 'escaping' cells in single-cell CRISPR screens
#           [Innovation]: 'escaping' cells
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: Ford hires 400 BlackBerry engineers for connected car development
#           [Innovation]: connected car development
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: BMW i Ventures invests in battery startup  ONE; cell-to-pack architecture.
#           [Innovation]: cell-to-pack architecture
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: Elon Musk & Jeff Bezos Have A Complicated History
#           [Innovation]: none
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: Solving the Plastic Shortage With an Efficient New Chemical Catalyst
#           [Innovation]: none
#           [Material]:  Chemical Catalyst
#           [utilization]: Plastic
#           ###
#           [Text]: Assessing a compound's activity, not just its structure, could accelerate drug discovery
#           [Innovation]: none
#           [Material]: compound's activity
#           [utilization]: drug discovery
#           ###
#           [Text]: Accuride Light-Weighting Continues With Two New Accu-LiteÂ® Steel Wheels
#           [Innovation]: Accu-LiteÂ® Steel Wheels
#           [Material]: none
#           [utilization]: none
#           ###
#           [Text]: %s
#           [Innovation]:""" % text,
#                                             max_length=1024,
#                                             end_sequence="\n###",
#                                             top_p=0.1,
#                                             remove_end_sequence=True,
#                                             remove_input=True)
#         list_innov.append(generation_tech['generated_text'])
#         print('The innovation term found is:', generation_tech['generated_text'])
#     return list_innov


def few_shot_open_ai(df):

    openai.api_key = "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"
    list_innov = []
    counter=1
    final_df = pd.DataFrame(columns=["Title", "Innovation"])
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
        data = {'Title': text,
                'Innovation': response["choices"][0]["text"]}
        df_temp = pd.DataFrame(data,index=[0])
        final_df = pd.concat([final_df, df_temp],ignore_index=True, axis=0)
        time.sleep(3)
        counter= counter+1
        print(counter)
        if counter%100 == 0:
            final_df.to_csv("./sample_files/GPT_output_data/newatlas_3000.csv")
    return final_df


def post_processing_nlp_cloud(few_shot_df: pd.DataFrame):
    few_shot_df['inno_wo_sp'] = few_shot_df['innov'].str.lower().replace('[{}]'.format(string.punctuation), '').replace(
        '[^\w\s]', '')
    few_shot_df['example_wo_sp'] = few_shot_df['Examples'].str.lower().replace('[^\w\s]', '')
    few_shot_df['inno_wo_sp'] = few_shot_df['inno_wo_sp'].map(lambda x: re.sub(r'[^\w\s]', '', x))
    few_shot_df['example_wo_sp'] = few_shot_df['example_wo_sp'].map(lambda x: re.sub(r'[^\w\s]', '', x))
    return few_shot_df


def post_processing_open_ai(few_shot_df: pd.DataFrame):
    df_temp = few_shot_df["Innov"].str.split("\n", expand=True)
    few_shot_df["innovation"] = df_temp[1]
    few_shot_df["material"] = df_temp[2]
    few_shot_df["utilization"] = df_temp[3]
    few_shot_df.drop(columns=["Innov"], inplace=True)
    few_shot_df.replace("\[[\w]+\] ?:", "", regex=True,inplace=True)
    return few_shot_df


def get_tokens(df):
    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
    df['sents_length'] = df.apply(lambda row: len(row['tokenized_sents']), axis=1)
    return df


if __name__ == "__main__":
    ### run gpt
    # path = "./sample_files/GPT_Input_Data/new_Atlas_3000.xlsx"
    # df = pd.read_excel(path)
    # final_df = few_shot_open_ai(df)

    ## postprocess gpt
    path = r"C:\Users\Brinda_Rao\Documents\Master_DKE\Fourth Sem\NER Project\Entity-Extraction-Using-NLP-Methods\named_entity_extraction\sample_files\GPT_output_data\new_Atlas.csv"
    df = pd.read_csv(path)
    df = post_processing_open_ai(df)
    df.to_csv(r"./sample_files/GPT_output_data/new_Atlas_postprocessed.csv")
    print(df)
