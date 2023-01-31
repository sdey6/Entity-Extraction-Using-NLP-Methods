import streamlit as st

import re
from spacy import displacy
from transformers import BertTokenizerFast
from named_entity_extraction.BERT.predict import predict_sent
import pickle
import spacy
import spacy_streamlit
from spacy.training import biluo_tags_to_offsets
from spacy.tokens import Doc
from transformers import RobertaTokenizerFast
from named_entity_extraction.Spacy.predict_evaluate.predict import entity_predictor

st.set_page_config(
    page_title="Named Entity Recognition Tagger", page_icon="📘"
)

ids_to_labels = {0: '-', 1: 'B-INNOVATION', 2: 'B-MATERIAL', 3: 'B-UTILIZATION', 4: 'I-INNOVATION', 5: 'I-MATERIAL',
                 6: 'I-UTILIZATION', 7: 'L-INNOVATION', 8: 'L-MATERIAL', 9: 'L-UTILIZATION', 10: 'O',
                 11: 'U-INNOVATION', 12: 'U-MATERIAL', 13: 'U-UTILIZATION'}

st.title("📘Named Entity Recognition Tagger")


######### App-related functions #########

@st.cache(allow_output_mutation=True)
def load_model(option):
    if option == "BERT":
        model = pickle.load(open("./STREAMLIT/Model/bert_model.pickle", "rb"))
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    else:
        model = pickle.load(open("./STREAMLIT/Model/roberta_model.pickle", "rb"))
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    nlp = spacy.load('en_core_web_sm')
    return model, tokenizer, nlp


def predict(sentence, model, tokenizer, nlp, ids_to_label):
    sentence, prediction = predict_sent(sentence, model, tokenizer, ids_to_label)

    doc = BILOU_to_offset(sentence, nlp, prediction)

    return doc


def BILOU_to_offset(sentence, nlp, prediction):
    class WhitespaceTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab

        def __call__(self, text):
            words = text.split(" ")
            spaces = [True] * len(words)
            # Avoid zero-length tokens
            for i, word in enumerate(words):
                if word == "":
                    words[i] = " "
                    spaces[i] = False
            # Remove the final trailing space
            if words[-1] == " ":
                words = words[0:-1]
                spaces = spaces[0:-1]
            else:
                spaces[-1] = False

            return Doc(self.vocab, words=words, spaces=spaces)

    ent_list = []

    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    doc = nlp(sentence)
    try:
        entities = biluo_tags_to_offsets(doc, prediction)
        for ent in entities:
            ent_dict = {"start": ent[0], "end": ent[1], "label": ent[2]}
            ent_list.append(ent_dict)
    except:
        st.error("No tags to be displayed")

    doc = {"text": sentence, "ents": ent_list, "title": None}

    return doc


# with st.form(key='my_form'):
#  option = st.selectbox(
#     'Choose a model below',
#    ('BERT',"ROBERTA",'Spacy'))

sentence = st.text_input(label='Enter a sentence:', max_chars=250)

model_names = ['BERT', "ROBERTA", 'SPACY']
model_option = st.radio("", model_names, index=0)
if model_option in ('BERT', "ROBERTA") and st.button('Predict'):
    if re.sub('\s+', '', sentence) == '':
        st.error('Please enter a non-empty sentence.')

    elif re.match(r'\A\s*\w+\s*\Z', sentence):
        st.error("Please enter a sentence with at least one word")

    else:

        model, tokenizer, nlp = load_model(model_option)

        result = predict(sentence, model, tokenizer, nlp, ids_to_labels)
        if result["ents"]:
            spacy_streamlit.visualize_ner(
                [result],
                labels=["INNOVATION", "MATERIAL", "UTILIZATION"],
                show_table=False,
                title="Predictions",
                manual=True,
                displacy_options={
                    "colors": {"INNOVATION": "#EFD6B0", "MATERIAL": "#E6C1CC", "UTILIZATION": "#C7E7F2"},
                    "kb_url_template": "https://www.wikidata.org/wiki/{}"
                },
                key="Custom Colors")
        else:
            st.error("No tags to be displayed")
elif model_option == 'SPACY' and st.button('Predict'):
    if re.sub('\s+', '', sentence) == '':
        st.error('Please enter a non-empty sentence.')

    elif re.match(r'\A\s*\w+\s*\Z', sentence):
        st.error("Please enter a sentence with at least one word")
    else:
        entity, doc = entity_predictor(sentence)
        # st.success(entity)
        ent_html = displacy.render(doc, style='ent', jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)

st.header("")
st.header("")
st.header("")
with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   The **Named Entity Recognition Tagger** app is a tool that performs named entity recognition.
-   The available entitites are: *innovation*, *material*, *utilization*, *organization*.
-   The app uses different models such as BERT,ROBERTA,SPACY,FLAIR fine-tuned on the mapegy dataset as well as scrapped datafrom web.            
       """ )
