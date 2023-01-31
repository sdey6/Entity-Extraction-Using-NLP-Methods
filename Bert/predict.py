from torch import cuda
import torch


def predict_sent(sentence,model,tokenizer, ids_to_labels):



    device = 'cuda' if cuda.is_available() else 'cpu'
    MAX_LEN = 128

    inputs = tokenizer(sentence,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=MAX_LEN,
                       return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask ,return_dict=False)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level


    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())

    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    previous_mapping =0
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        # only predictions on first word pieces are important
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
            previous_mapping = mapping[1]
        elif mapping[0] != previous_mapping:
            prediction.append(token_pred[1])
            previous_mapping = mapping[1]

        else:
            previous_mapping = mapping[1]
            continue

    print(sentence)
    print(prediction[:-1])

    final_predictions = prediction[:-1]

    # convert BILOU tags to BILO tags
    for pos, pred in enumerate(final_predictions):
        if pred.startswith("B"):
            if not ((final_predictions[pos + 1].startswith("I") or final_predictions[pos + 1].startswith("L"))):
                final_predictions[pos] = pred.replace("B", "U")

        if pred.startswith("I"):
            if not ((final_predictions[pos - 1].startswith("B")) or (final_predictions[pos - 1].startswith("I"))):
                final_predictions[pos] = pred.replace("I-", "U-")
            if ((final_predictions[pos - 1].startswith("B")) and final_predictions[pos -1].startswith("I")) or (final_predictions[pos + 1].startswith("O")):
                final_predictions[pos] = pred.replace("I-", "L-")


        if pred.startswith("L"):
            if not ((final_predictions[pos - 1].startswith("B")) or (final_predictions[pos - 1].startswith("I"))):
                final_predictions[pos] = pred.replace("L", "U")

    return sentence, final_predictions








