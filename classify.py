from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.metrics import classification_report


trainfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT-out-CONLL.conll"
testfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-test-cardboard.txt"

vec = DictVectorizer()

all_rows_features = []
all_rows_labels = []

def get_features(inputfilepath):

    with open(inputfilepath, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

        for line in lines:
            row = line.rstrip("\n").split("\t")

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
                print(row_features)
                all_rows_features.append(row_features)
                all_rows_labels.append(label)

    return all_rows_features, all_rows_labels

    print("len features", len(all_rows_features))
    print("len labels", len(all_rows_labels))


train_features, train_labels = get_features(trainfile_path)
train_vector_features = vec.fit_transform(train_features)
print("train features vector shape", train_vector_features.shape)

# Change model to test others
model = svm.LinearSVC()
model.fit(train_vector_features, train_labels)

print("----------------------------------------------------TEST-------------------------------------------------------")
test_features, test_labels = get_features(testfile_path)
test_vector_features = vec.transform(test_features)
print("test features vector shape", test_vector_features.shape)

predictions = model.predict(test_vector_features)


print(classification_report(test_labels, predictions))
