from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


trainfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT-out-CONLL.conll"
testfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-test-cardboard.txt"

vec = DictVectorizer()

def get_features(inputfilepath):
    all_rows_features = []
    all_rows_labels = []

    with open(inputfilepath, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        print(len(lines))

        for line in lines:
            row = line.rstrip("\n").split("\t")
            #print(row)

            if len(row) > 1:
                filename = row[0]
                sentence_nr = row[1]
                token_nr = row[2]
                word = row[3]
                label = row[4]
                """
                pos = row[5]
                lemma = row[6]
                pos_category = row[7]
                sentiment = row[8]
                is_expression = row[9]
                normalised_position = row[10]
                has_prefix = row[11]
                has_suffix = row[12]
                is_antonym = row[13]
                previous_pos = row[14]
                next_pos = row[15]
                previous_lemma = row[16]
                next_lemma = row[17]
                """

                row_features = {
                    "token": word
                    #,"POS": pos
                    #,"lemma": lemma
                    #,"POS_category": pos_category
                    #,"sentiment": sentiment
                    #,"isExpression": is_expression
                    #,"normalised_position": normalised_position
                    #,"hasPrefix": has_prefix
                    #,"hasSuffix": has_suffix
                    #,"isAntonym": is_antonym
                    #,"previous_POS": previous_pos
                    #,"next_POS": next_pos
                    #,"previous_lemma": previous_lemma
                    #,"next_lemma": next_lemma
                }
                #print(row_features)
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

print("----------------------------------------------------TEST-------------------------------------------------------")
test_features, test_labels = get_features(testfile_path)
test_vector_features = vec.transform(test_features)
print("test features vector shape", test_vector_features.shape)

find_I_negs(train_labels)

predictions = model.predict(test_vector_features)

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

print(classification_report(test_labels, predictions))
