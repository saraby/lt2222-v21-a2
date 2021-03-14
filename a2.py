from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix as cm
from collections import Counter
from nltk.corpus import wordnet as wn
import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
import string
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')


# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1


def preprocess(inputfile):
    wnl = WordNetLemmatizer()
    tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    columns_names = ["Line_No", "Sentence_No", "Word", "POS", "NER"]
    text_list = []
    # freq_words=[]
    # chosen_words=[]

    for row in inputfile.readlines()[1:]:
        row_items = row.rstrip().split("\t")
        row_items[0] = int(row_items[0])
        row_items[1] = int(float(row_items[1]))
        if row_items[3].isalpha():
            if row_items[3] not in tags:
                row_items[2] = wnl.lemmatize(row_items[2].lower())
            else:
                row_items[2] = wn._morphy(row_items[2].lower(), pos='v')
                verb = str(row_items[2]).strip("['").strip("']")
                row_items[2] = verb
            # freq_words.append(row_items[2])
            text_list.append(row_items)
    # most_freq = Counter(freq_words).most_common()
    # chosen_words = [i for (i,j) in most_freq if j > 1]
    # # print(chosen_words[4:10])
    # for row in text_list:
    #     if row[2] not in chosen_words:
    #         text_list.pop(text_list.index(row))
    fd = pd.DataFrame(text_list, columns=columns_names)
    # fd.set_index("Line_No")
    return fd

# Code for part 2


class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)


def create_instances(data):
    data_list = data.values.tolist()

    savior_list = [[0, 0, "", "", ""]]*5
    for l in savior_list:
        data_list.append(l)
        data_list.insert(0, l)

    instances = []
    for i, sent in enumerate(data_list):
        features = []
        pre_features = []
        post_features = []
        start_tokens = ["<S5>", "<S4>", "<S3>", "<S2>", "<S1>"]
        end_tokens = ["</S5>", "</S4>", "</S3>", "</S2>", "</S1>"]

        index = i + 5
        if sent[4].startswith("B"):

            neclass = sent[4][2:]
            for n in range(1, 6):
                pre_row = data_list[index-n]
                embedded_word = embedd(
                    sent[1], pre_row[1], pre_row[4], pre_row[2], start_tokens)
                pre_features.insert(0, embedded_word)

            skipped = 1
            while data_list[index+skipped][4].startswith('I'):
                skipped = skipped+1
            for m in range(1, 6):
                if (len(data_list)-index) >= 5:
                    post_row = data_list[index+skipped+m-1]
                    embedded_word = embedd(
                        sent[1], post_row[1], post_row[4], post_row[2], end_tokens)
                    post_features.append(embedded_word)

            features = pre_features + post_features

            instances.append(Instance(neclass, features))
    return instances


def embedd(sent_no, feat_sent_no, ner, word, tokens):
    if sent_no == feat_sent_no and ner == "O" and len(tokens) == 5:
        return word
    else:
        return tokens.pop(0)

# Code for part 3


def create_table(instances):

    extracted_features = [
        feature for instance in instances for feature in instance.features]
    most_freq = Counter(extracted_features).most_common(3000)
    top_freq = [i for (i, j) in most_freq]
    cols = ["Class"] + top_freq
    rows = [[instance.neclass]+[feature.count(word) for word in top_freq]
            for instance in instances for feature in instance.features]

    df = pd.DataFrame(rows, columns=cols)
    reducted_df = reduce(df.drop('Class', 1))
    reducted_df.insert(0, "Class", df["Class"])

    return reducted_df


def reduce(matrix, dims=300):
    svd = TruncatedSVD(n_components=dims)
    matrix_reduced = svd.fit_transform(matrix)
    matrix_df = pd.DataFrame(matrix_reduced, index=matrix.index)
    return matrix_df


def ttsplit(bigdf):

    df_train = bigdf.sample(frac=0.8).reset_index()
    df_test = bigdf.drop(df_train.index).reset_index()

    return df_train.drop('Class', axis=1).to_numpy(), df_train['Class'], df_test.drop('Class', axis=1).to_numpy(), df_test['Class']

# Code for part 5


def confusion_matrix(truth, predictions):

    labels = list(np.unique(truth.T))
    confusion_matrix = cm(truth, predictions, labels=labels)
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    return df


# Code for bonus part B
def bonusb(filename):
    pass
