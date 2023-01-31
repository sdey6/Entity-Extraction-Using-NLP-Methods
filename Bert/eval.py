from dataset import dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score

def evaluate(model, tokenizer, labels_to_ids,ids_to_labels,test_dataset):
    # put model in evaluation mode

    VALID_BATCH_SIZE = 2
    MAX_LEN = 128

    model.eval()


    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 1
                   }

    training_set = dataset(test_dataset, tokenizer, MAX_LEN, labels_to_ids)
    testing_loader = DataLoader(training_set, **test_params)

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                with open("./Output_Files/roberta_output.txt", "a") as myfile:
                    myfile.write(f"Validation loss per 100 evaluation steps: {loss_step}\n")

                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    with open("./Output_Files/output.txt", "a") as myfile:
        myfile.write(f"Validation Loss: {eval_loss}\n")
        myfile.write(f"Validation Accuracy: {eval_accuracy}\n")
        myfile.close()

    labels = [labels]
    predictions = [predictions]

    return model, labels, predictions