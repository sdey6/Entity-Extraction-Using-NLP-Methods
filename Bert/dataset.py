import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import Dataset


def preprocess(df):

    labels = []
    for _, rows in df.iterrows():
        row = literal_eval(rows["BIO"])
        labels.append(row)
    df["BIO"] = labels
    return df


def read_data():

    df = pd.read_csv(r"./Input_Files/final_post_processed.csv")
    df = df.drop(columns=["Unnamed: 0"])

    df = preprocess(df)

    labels = []
    for _, rows in df.iterrows():
        labels.append(rows["BIO"])

    # Check how many labels are there in the dataset
    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    train_dataset, test_dataset= split_data(df)

    return unique_labels,labels_to_ids,ids_to_labels,train_dataset,test_dataset

def split_data(df):

    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    return train_dataset,test_dataset

def align_label(texts, labels,tokenizer,labels_to_ids):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=128, truncation=True)
    label_all_tokens = False
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.title[index]

        word_labels = self.data.BIO[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        encoded_labels = [align_label(self.data.title[index], word_labels, self.tokenizer, self.labels_to_ids)]

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len