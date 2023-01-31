# import evaluate_spacy
# import prepare_training_data
# from config import path
from named_entity_extraction.Spacy.predict_evaluate.predict import load_model
from named_entity_extraction.Spacy.Model_Training import prepare_train_data
# from evaluate_spacy import get_score
from named_entity_extraction.Spacy.predict_evaluate.evaluate import inno_list, pred_list, example_list
from named_entity_extraction.Spacy.predict_evaluate import evaluate
from named_entity_extraction.Config.config import pickle_spacy_path

nlp = load_model(pickle_spacy_path)
print('Using spacy pickled trained model print some training output..........')
for text, _ in prepare_train_data.data_training[:10]:
    doc = nlp(text)
    print('entities:', [(ent.text, ent.label_) for ent in doc.ents])

print('Getting scores on training.........')
print(evaluate.get_score(prepare_train_data.df_spacy_pp))
print('Data prepared for evaluating on unseen data.....')
print(f'Length is:{len(inno_list)},{len(pred_list)},{len(example_list)}')
print(inno_list[:5], pred_list[:5], example_list[:5])
print('get MUC score from test data........')
print(evaluate.get_predictive_score(inno_list, pred_list,example_list))
print(f'Overall flat accuracy is:{evaluate.accuracy}%')
print(f'Token-wise mean accuracy:{evaluate.mean_acc}')
