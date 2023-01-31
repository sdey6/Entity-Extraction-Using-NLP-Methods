import pandas as pd
import re
import spacy
from spacy.training import offsets_to_biluo_tags
import numpy as np


def clean(text):
    """
    Just a helper fuction to add a space before the punctuations for better tokenization
    """
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'", ]
    try:
        for i in text:
            if i in filters:
                text = text.replace(i, "")
    except:
        print(text)
    return text


def preprocessing_tags(df):
    for col in df.columns:
        df[col] = df[col].map(lambda x: clean(x))
    # todo needs to be implemented in postprocessing of gpt
    df.replace("[ ]{2,}", "", regex=True, inplace=True)
    regex = re.compile(r'(none|nan|NaN|^\s*$)')
    df = df.replace(regex, "", regex=True)
    df = df.apply(lambda x: x.astype(str).str.lower())

    # create a tags column combining all the columns
    tags = []
    for i, j in df.iterrows():
        tag_list = []
        if j["innovation"].__len__() != 0:
            tag_list.append((j["innovation"], "INNOVATION"))
        if j["material"].__len__() != 0:
            tag_list.append((j["material"], "MATERIAL"))
        if j["utilization"].__len__() != 0:
            tag_list.append((j["utilization"], "UTILIZATION"))
        if tag_list:
            tags.append(tag_list)
        else:
            tags.append("")
    df["annotation"] = tags

    # drop innovation, material and utilizaton columns
    df = df.drop(columns=["innovation", "material", "utilization", ])
    df = df[df[["annotation"]].ne('').sum(1).ge(1)]
    return df


def create_fomat(df):
    ################################################# create spacy format ##############################################

    annotations = []
    for i, rows in df.iterrows():
        annot_list = []
        prev_pos = []
        for span in rows["annotation"]:
            for match in re.finditer(span[0].strip(), rows["title"].strip()):
                pos = match.span()

                ## to check ovelapping entities
                if annot_list:
                    if pos not in prev_pos:
                        annot_tup = pos + (span[1],)
                        annot_list.append(annot_tup)
                else:
                    annot_tup = pos + (span[1],)
                    annot_list.append(annot_tup)
                prev_pos.append(pos)

        if annot_list:
            annotations.append(annot_list)
        else:
            annotations.append("")
    df["spacy"] = annotations

    ################################################ create BIO format ##################################################

    nlp = spacy.load('en_core_web_sm')
    BIO_tags = []
    for _, rows in df.iterrows():
        try:
            doc = nlp.make_doc(rows["title"])
            entities = rows["spacy"]
            tags = offsets_to_biluo_tags(doc, entities)
            if tags:
                BIO_tags.append(tags)
            else:
                BIO_tags.append("")
        except:
            BIO_tags.append("")
    df["BIO"] = BIO_tags

    ## removing empty rows
    df = df.replace("", np.NAN, )
    df = df.dropna(axis=0)
    return df


def main():
    path = "./GPT/Files/few_shot_post_processed.csv"
    df_mapegy = pd.read_csv(path)
    path = "./GPT/Files/new_Atlas_postprocessed.csv"
    df_newatlas = pd.read_csv(path)
    final_df = pd.concat([df_mapegy, df_newatlas], ignore_index=True, axis=0)
    final_df = preprocessing_tags(final_df)
    final_df = create_fomat(final_df)
    final_df.to_csv(r"./GPT/Files/final_post_processed.csv")
    # print(final_df)


if __name__ == '__main__':
    main()
