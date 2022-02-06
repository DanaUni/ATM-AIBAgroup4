import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
import distutils.util


trainfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT-out-CONLL.conll"
testfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-test-cardboard.txt"

vec = DictVectorizer()

def get_features(inputfilepath):
    """
    
    """
    all_rows_features = []
    all_rows_labels = []

    with open(inputfilepath, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        print(len(lines))

        for line in lines:
            row = line.rstrip("\n").split("\t")

            if len(row) > 1:
               filename = row[0]
                sentence_nr = row[1]
                token_nr = row[2]
                word = row[3]
                label = row[4]
                pos = row[5]
                lemma = row[6]
                pos_category = row[7]
                is_expression = row[8]
                has_prefix = row[9]
                has_suffix = row[10]
                is_antonym = row[11]
                previous_lemma = row[12]
                next_lemma = row[13]
                previous_pos = row[14]
                next_pos = row[15]
                norm_position_binned = row[16]
                sentiment_category = row[17]
                
                is_expression = bool(distutils.util.strtobool(row[8]))
                has_prefix = bool(distutils.util.strtobool(row[9]))
                has_suffix = bool(distutils.util.strtobool(row[10]))
                is_antonym = bool(distutils.util.strtobool(row[11]))

                row_features = {
                    "token": word
                    , "hasPrefix": has_prefix
                    , "hasSuffix": has_suffix
                    , "isAntonym": is_antonym
                    , "POS": pos
                    , "lemma": lemma
                    , "POS_category": pos_category
                    , "isExpression": is_expression
                    , "previous_POS": previous_pos
                    , "next_POS": next_pos
                    , "previous_lemma": previous_lemma
                    , "next_lemma": next_lemma
                    , "normalised_position_binned": norm_position_binned
                    , "sentiment_category": sentiment_category
                }
                
                all_rows_features.append(row_features)
                all_rows_labels.append(label)

    return all_rows_features, all_rows_labels

    print("len features", len(all_rows_features))
    print("len labels", len(all_rows_labels))


train_features, train_labels = get_features(trainfile_path)
train_vector_features = vec.fit_transform(train_features)
print("train features vector shape", train_vector_features.shape)

def find_I_negs(listy):
    count_inegs = 0
    for itemm in listy:
        if itemm == "I-NEG":
            count_inegs += 1
    print("number of I-NEGS", count_inegs)

find_I_negs(train_labels)

# Change model to test others
model = svm.LinearSVC()
model.fit(train_vector_features, train_labels)

#### Grid search for MLP
# parameter_space = {
#     'hidden_layer_sizes': [(10,10,10), (20,20,20), (10,10), (20,20), (10,10, 10, 10), (20, 20, 20, 20)],
#     'activation': ['tanh', 'relu', 'logistic'],
#     'learning_rate_init': [0.01, 0.001, 0.0005]}


# model = MLPClassifier(hidden_layer_sizes=(20,20, 20))
# model.fit(train_vector_features, train_labels)

#### Enable lines below for gridsearch
# mlp = MLPClassifier(max_iter=100)
# clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
# clf.fit(train_vector_features, train_labels)
# print('Best parameters found:\n', clf.best_params_)




print("----------------------------------------------------TEST-------------------------------------------------------")
test_features, test_labels = get_features(testfile_path)
test_vector_features = vec.transform(test_features)
print("test features vector shape", test_vector_features.shape)

find_I_negs(train_labels)

predictions = model.predict(test_vector_features)

#### Enable for grid search
#predictions = clf.predict(test_vector_features)

print("\nB-NEG false negatives [prediction] [gold] [token]")
for i in range(len(test_labels)):
    if (test_labels[i] == "B-NEG") and (predictions[i] != "B-NEG"):
        print(predictions[i], test_labels[i], test_features[i]["token"])
print()

print("\nI-NEG predictions [prediction] [gold] [token]")
for i in range(len(predictions)):
    if predictions[i] == "I-NEG":
        print(predictions[i], test_labels[i], test_features[i])

count_bnegs = 0
for i in range(len(predictions)):
    if test_labels[i] == "I-NEG":
        count_bnegs += 1
        print(predictions[i], test_labels[i], test_features[i])

print(count_bnegs)

for test_label in test_labels:
    if test_label == "I-NEG":
        print("yes")
        
print("i prediction gold token")
for i in range(len(test_labels)):
    if test_labels[i] != predictions[i]:
        print(i, predictions[i], test_labels[i], test_features[i]["token"])

print(classification_report(test_labels, predictions, digits=3))
print("Full confusion matrix (all classes)\n", confusion_matrix(test_labels, predictions), "\n")
print("Confusion matrix without class O\n", confusion_matrix(test_labels, predictions, labels=["B-NEG", "I-NEG"]))
tn, fp, fn, tp = confusion_matrix(test_labels, predictions, labels=["B-NEG",  "I-NEG"]).ravel()
print()
prfs = precision_recall_fscore_support(test_labels, predictions, labels=["B-NEG",  "I-NEG"])
prfs_micro_avg = precision_recall_fscore_support(test_labels, predictions, average="micro", labels=["B-NEG",  "I-NEG"])

print("Without class O")
print("       Precision Recall F-score support")
print(f"B-NEG: {round(prfs[0][0], 3)}     {round(prfs[1][0], 3)}  {round(prfs[2][0], 3)}   {round(prfs[3][0], 3)}")
print(f"I-NEG: {round(prfs[0][1], 3)}       {round(prfs[1][1], 3)}  {round(prfs[2][1], 3)}     {round(prfs[3][1], 3)}")
print()

print("F1-scores")
print(f"B-NEG:         {round(prfs[2][0], 3)}")
print(f"I-NEG:         {round(prfs[2][1], 3)}")
print(f"Micro average: {round(prfs_micro_avg[2], 3)}")
