from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

#lemmatizer = WordNetLemmatizer()

lines = []
current_sentence = []
all_sentences = []
pos_tagged_sentences = []
data = []

with open("..\data\SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt", "r") as infile:
    for line in infile:
        line = line.strip().split()

        # If not an empty line
        if len(line) > 0:
            filename = line[0]
            sentence_nr = line[1]
            token_nr = line[2]
            word = line[3]
            label = line[4]

            current_sentence.append(word)

        else:
            #print("\nempty\n")
            pos_tagged_sent = pos_tag(current_sentence)
            pos_tagged_sentences.append(pos_tagged_sent)

            all_sentences.append(current_sentence)
            current_sentence = []
            #print(pos_tagged_sent)

for sentence in pos_tagged_sentences:
    for token, pos_tag in sentence:
        data.append({'token': token, 'pos_tag': pos_tag})


print("number of sentences", len(all_sentences))
print(data)

