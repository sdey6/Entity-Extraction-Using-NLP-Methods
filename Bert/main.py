from transformers import BertTokenizerFast, BertForTokenClassification,RobertaForTokenClassification,RobertaTokenizerFast
from dataset import read_data, align_label
from model import  train_loop
from eval import evaluate
import pickle
from torch import cuda
from seqeval.metrics import classification_report



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
    unique_labels, labels_to_ids, ids_to_labels, train_dataset, test_dataset = read_data()
    model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=len(labels_to_ids))
    model.to(device)
    model = train_loop(model, tokenizer, labels_to_ids, train_dataset, test_dataset)
    model, labels, predictions = evaluate(model, tokenizer, labels_to_ids, ids_to_labels, test_dataset)

    print(classification_report(labels, predictions))




    filename = f"""./Output_Files/roberta_model_predict.pickle"""
    pickle.dump(model, open(filename, "wb"))


    filename = f"""./Output_Files/roberta_predictions.pickle"""
    pickle.dump(model, open(filename, "wb"))

    filename = f"""./Output_Files/roberta_labels.pickle"""
    pickle.dump(model, open(filename, "wb"))

    with open("./Output_Files/roberta_output.txt", "a") as myfile:
        myfile.write(classification_report(labels, predictions))
        myfile.close()

    # filename = "./named_entity_extraction/BERT/ner_model.pickle"
    # pickle.dump(model, open(filename, "wb"))


