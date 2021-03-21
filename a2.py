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
    tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # all verb tags
    # the labels of the dataframe which we'll build
    columns_names = ["Line_No", "Sentence_No", "Word", "POS", "NER"]
    text_list = []  # the list which will include all the selected and filtered data

    # reading the lines of the file except for the first one because it has the labels
    for row in inputfile.readlines()[1:]:
        row_items = row.rstrip().split("\t")  # remove the new lines and the tabs
        # convert the string line numbers into integers
        row_items[0] = int(row_items[0])
        # convert the string float sentence numbers into integers
        row_items[1] = int(float(row_items[1]))
        if row_items[3].isalpha():  # remove all the words which has no recognised pos that includes punctuations and words like """" which I happened to see in the output
            if row_items[3] not in tags:  # if the word is not a verb use the wnl lemmatizer
                row_items[2] = wnl.lemmatize(row_items[2].lower())
            else:
                # if it is a verb then use another lemmatizer because the previous one did not convert the verbs dealing with it as if they are nouns
                row_items[2] = wn._morphy(row_items[2].lower(), pos='v')
                # when using the last lemmatizer it will add the lemma to a list so we need remove the list and quotations
                verb = str(row_items[2]).strip("['").strip("']")
                row_items[2] = verb
            # append the row items after modifications
            text_list.append(row_items)

    # create the dataframe for the file
    fd = pd.DataFrame(text_list, columns=columns_names)

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
    # convert the dataframe back to a list to make it easier to deal with
    data_list = data.values.tolist()

# I am using a 5 temporary rows before and after the original rows so we can avoid the problem when processing the last and first 5 words
    savior_list = [[0, 0, "", "", ""]]*5
    for l in savior_list:
        data_list.append(l)
        data_list.insert(0, l)

    instances = []
    for i, sent in enumerate(data_list):  # use enumerate to control indexing
        features = []  # list of all 10 the features
        pre_features = []  # list of all 5 features before the ne word
        post_features = []  # list of all 5 features after the ne word
        start_tokens = ["<S5>", "<S4>", "<S3>", "<S2>", "<S1>"]
        end_tokens = ["</S5>", "</S4>", "</S3>", "</S2>", "</S1>"]

        index = i + 5
        # if the word's named entity starts with 'B' then start getting the features as follows:
        if sent[4].startswith("B"):

            neclass = sent[4][2:]  # extract the neclass (last 3 characters)
            for n in range(1, 6):  # start considering the first 5 features
                # getting the last previous 5 features
                pre_row = data_list[index-n]
                # calling out embedd function which will give us the feature to add it to the list
                embedded_word = embedd(
                    sent[1], pre_row[1], pre_row[4], pre_row[2], start_tokens)
                # add the extracted feature as the first item of the list
                pre_features.insert(0, embedded_word)

            skipped = 1  # initialising a counter to add 1 when there is an named entity starts with an 'I'
            while data_list[index+skipped][4].startswith("I"):
                skipped = skipped+1
            for m in range(1, 6):  # start considering the second 5 features
                if (len(data_list)-index) >= 5:
                    # getting the first last 5 features
                    post_row = data_list[index+skipped+m-1]
                    embedded_word = embedd(
                        sent[1], post_row[1], post_row[4], post_row[2], end_tokens)
                    # add the extracted feature as the last item of the list
                    post_features.append(embedded_word)

            features = pre_features + post_features  # join all two features lists into one

            instances.append(Instance(neclass, features))
    return instances

# a function to choose which word to embedd into the features


def embedd(sent_no, feat_sent_no, ner, word, tokens):
    ''' 
    if the processed word was in the same sentence as the ne and it doesn't begin with 'B' or 'I'
    and the tokens were still untouched then add the word to the feature 
    but if it's in another sentence or is another named entity or has been preceded by a token then add all the left tokens
    '''
    if sent_no == feat_sent_no and ner == "O" and len(tokens) == 5:
        return word
    else:
        return tokens.pop(0)

# Code for part 3


def create_table(instances):

    # extract all features from each class and add them to big list
    extracted_features = [
        feature for instance in instances for feature in instance.features]
    # order the features by the number of occurrences
    most_freq = Counter(extracted_features).most_common(3000)
    # take only the ordered features w=and ignore number of occurrences
    top_freq = [i for (i, j) in most_freq]
    cols = ["Class"] + top_freq  # initiate the names of the columns
    rows = [[instance.neclass]+[feature.count(word) for word in top_freq]  # start creating the rows by adding the neclass as the first item
            for instance in instances for feature in instance.features]  # then the number of occurrences of each feature in each class

    # create the dataframe and reduce the matrix dimensionality to 300 dimensions to improve the model and results
    df = pd.DataFrame(rows, columns=cols)
    # drop the first column and then add it after reduction
    reduced_df = reduce(df.drop('Class', 1))
    reduced_df.insert(0, "Class", df["Class"])

    return reduced_df

# reduction function from assignment 1


def reduce(matrix, dims=300):
    svd = TruncatedSVD(n_components=dims)
    matrix_reduced = svd.fit_transform(matrix)
    matrix_df = pd.DataFrame(matrix_reduced, index=matrix.index)
    return matrix_df


def ttsplit(bigdf):
    # take 80% of the data randomly as training data and then reset the index so the old indexes doesn't show the randomization
    # then take the rest of the data bu dropping all rows belonging to the trining data from the dataframe and reset the index too
    train = bigdf.sample(frac=0.8)
    df_train = train.reset_index()
    df_test = bigdf.drop(train.index).reset_index()

    return df_train.drop('Class', axis=1).to_numpy(), df_train['Class'], df_test.drop('Class', axis=1).to_numpy(), df_test['Class']

# Code for part 5


def confusion_matrix(truth, predictions):
    # take all the classes without repeating and use them as labels

    # labels = list(np.unique(truth.T))
    labels = list(np.unique([truth.T] + [predictions.T]))
    # use the confusion matrix by the sklearn
    confusion_matrix = cm(truth, predictions, labels=labels)
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    return df


# Code for bonus part B
def bonusb(filename):
    pass
