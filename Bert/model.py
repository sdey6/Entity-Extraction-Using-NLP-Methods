import torch
from transformers import BertTokenizerFast, BertForTokenClassification,RobertaTokenizerFast
from dataset import dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pickle


def train_loop(model, tokenizer, labels_to_ids, train_dataset, test_dataset):


    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    EPOCHS = 15
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    training_set = dataset(train_dataset, tokenizer, MAX_LEN,labels_to_ids)
    training_loader = DataLoader(training_set, **train_params)

    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        with open("./Output_Files/roberta_output.txt", "a") as myfile:
            myfile.write(f"Training epoch: {epoch + 1}")

        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        model.train()

        for idx, batch in enumerate(training_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}\n")

                with open("./Output_Files/roberta_output.txt", "a") as myfile:
                    myfile.write(f"Training loss per 100 training steps: {loss_step}\n")


            # compute training accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
            # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}\n")
        print(f"Training accuracy epoch: {tr_accuracy}\n")

        with open("./Output_Files/roberta_output.txt", "a") as myfile:
            myfile.write(f"Training loss epoch: {epoch_loss}\n")
            myfile.write(f"Training accuracy epoch: {tr_accuracy}\n")
            myfile.close()

        filename = f"""./Output_Files/roberta_model_train.pickle"""
        pickle.dump(model, open(filename, "wb"))

    return model
