import pickle
from named_entity_extraction.Config.config import pickle_spacy_path


def load_model(path):
    spacy_model = pickle.load(open(path, 'rb'))
    return spacy_model


def entity_predictor(text):
    nlp = load_model(pickle_spacy_path)
    docx = nlp(text)
    pred_entity = [(ent.text, ent.label_) for ent in docx.ents]
    return pred_entity, docx

