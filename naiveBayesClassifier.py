import math
import numpy as np
import pandas as pd

"""
Reads all documents from the
train set and returns the following data structures
Params:file_name
Return:
-texall - list of documents; each entry corresponds to a document which is list of words.
-lbAll list of documents' labels.
-voc - set of all distinct words in the train set.
-cat - set of document categories.
"""


def readTrainData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    texAll = []
    lbAll = []
    voc = []
    for line in lines:
        splitted = line.split('\t')
        lbAll.append(splitted[0])
        texAll.append(splitted[1].split())
        words = splitted[1].split()
        for w in words:
            voc.append(w)
    voc = set(voc)
    cat = set(lbAll)
    return texAll, lbAll, voc, cat


texAll, lbAll, voc, cat = readTrainData('r8-train-stemmed.txt')

"""
computes and returns the probabilities
Return:
-Pw - a matrix of class-conditional probabilities, p(x|wi)
-P - a vector of class priors, p(wi)
"""


def learn_NB_text():
    # Calculate a vector of class priors
    total = len(lbAll)
    p = [lbAll.count(category) / total for category in cat]
    p = pd.DataFrame([p], index=["p(wi)"], columns=cat)
    # Calculate a matrix of class-conditional probabilities
    voc.add('NA')  # Add NA for thw words that present in test but not in train set
    pw = pd.DataFrame(0, index=voc, columns=cat)  # Initiate data frame with zeros in size vocXcat
    for sentence, category in zip(texAll, lbAll):  # Count how many times a word appears in each category
        for word in sentence:
            pw[category][word] += 1
    # Change the dataframe to contain  p(x|wi) with Laplace smoothing (add 1 to the count and add the number of classes to the denominator)
    col_sum = pw.sum(axis=0) + pw.shape[0]
    pw = (pw + 1).div(col_sum, axis=1)
    return pw, p


pw, p = learn_NB_text()

"""
Example for the outputs
p:
           earn     ship       acq  ...  interest     grain     trade
p(wi)  0.517776  0.01969  0.290975  ...   0.03464  0.007475  0.045761

[1 rows x 8 columns]
pw:
                 earn      ship       acq  ...  interest     grain     trade
enro         0.000007  0.000041  0.000091  ...  0.000036  0.000052  0.000021
mdi          0.000013  0.000041  0.000008  ...  0.000036  0.000052  0.000021
aciv         0.000013  0.000041  0.000008  ...  0.000036  0.000052  0.000021
imprecis     0.000013  0.000041  0.000008  ...  0.000036  0.000052  0.000021
fomc         0.000007  0.000041  0.000008  ...  0.000107  0.000052  0.000021
...               ...       ...       ...  ...       ...       ...       ...
countervail  0.000007  0.000041  0.000008  ...  0.000036  0.000052  0.000062
cubic        0.000027  0.000082  0.000038  ...  0.000036  0.000052  0.000021
shepard      0.000007  0.000041  0.000015  ...  0.000036  0.000052  0.000021
wimi         0.000013  0.000041  0.000008  ...  0.000036  0.000052  0.000021
district     0.000027  0.000041  0.000137  ...  0.000036  0.000052  0.000041

[14575 rows x 8 columns]

"""
texAll_test, lbAll_test, _test, cat_test = readTrainData('r8-test-stemmed.txt')
"""
classifies all documents from the test set and computes the success rate (suc) as a
number of correctly classified documents divided by the number of all
documents in the test set.
Params:
-Pw - a matrix of class-conditional probabilities, p(x|wi)
-P - a vector of class priors, p(wi)
Return:
-suc - success rate
"""


def ClassifyNB_text(pw, p):
    all_documents = len(texAll_test)
    correctly_classified_documents = 0
    for sentence, label in zip(texAll_test, lbAll_test):
        most_probable_category = ''
        most_probable_category_p = -float('inf')
        for category in cat:
            p_words_in_sentence = [(pw[category][word] if word in voc else pw[category]['NA']) for word in
                                   sentence]  # Array of probability of a word in a category
            p_sentence_category = np.log(
                p_words_in_sentence).sum()  # Summing the logs of probabilities (instead of multiplying probabilities)
            p_sentence_category += math.log(p[category])  # summing the log of the category prior
            if p_sentence_category > most_probable_category_p:
                most_probable_category_p = p_sentence_category
                most_probable_category = category
        if most_probable_category == label:
            correctly_classified_documents += 1
    return correctly_classified_documents / all_documents


suc = ClassifyNB_text(pw, p)
print(suc)
# classification rate = 0.9643672910004568
