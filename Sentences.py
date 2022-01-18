from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Only run this the first time
#nltk.download("wordnet")

infile_path = "..\data\SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt"
outfile_path = infile_path.replace(".txt", "-out.txt")
lemmatizer = WordNetLemmatizer()
current_sentence = []
current_sentence_dict = dict()
data = []

def get_wordnet_pos(pos_tag):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(pos_tag, wordnet.NOUN)

def extract_and_add_sentence_features(sentences: dict, dataframe):
    """
    Extract features from the dataset that depend on the context/sentence and add these features to the dataframe.

    :param sentences: dictionary of a list of each sentence from the dataset (value) and their sentence id (key)
    :param dataframe: pandas dataframe that that contains the dataset
    """

    print("in extract_and_add_features")
    dataframe["POS"] = ""
    dataframe["lemma"] = ""
    sentence_features = dict()

    for sid, sentence in sentences.items():
        print("SID:", sid)

        # POS-tag the sentence
        pos_tagged_sent = pos_tag(sentence)

        # Using SpaCy for lemmatisation
        #sentence_string = " ".join(sentence)
        #lemma_sent = nlp(sentence_string)

        # Add sentence based features to a dictionary
        #sentence_features[sid] = {"POS": pos_tagged_sent, "lemma": lemma_sent}
        sentence_features[sid] = {"POS": pos_tagged_sent}

        # Add sentence based features to dataframe
        for token_index in range(len(sentence)):
            indices = dataframe.index[(dataframe["sent_id"] == sid) & (dataframe["token_nr"] == str(token_index))].to_list()
            for row_index in indices:
                dataframe.loc[row_index, "POS"] = sentence_features[sid]["POS"][token_index][1] #[1] for the pos tag in the (token, pos) tuple

                # Using NLTK Wordnet for lemmatisation
                current_token = sentence_features[sid]["POS"][token_index][0]
                pos = get_wordnet_pos(sentence_features[sid]["POS"][token_index][1])
                lemma = lemmatizer.lemmatize(current_token, pos)
                dataframe.loc[row_index, "lemma"] = lemma

                # Using SpaCy for lemmatisation
                #dataframe.loc[row_index, "lemma"] = sentence_features[sid]["lemma"][token_index].lemma_


def contains_prefix(word):
    #List can and probably should be extended (comment: we also need a source for the list or use an existing one)
    prefix_list = ['in', 'un', 'im', 'il', 'dis']
    if word.startswith(tuple(prefix_list)):
        return True
    else:
        return False

def contains_suffix(word):
    #List can and probably should be extended (comment: we also need a source for the list or use an existing one)
    suffix_list = ['less']
    if word.endswith(tuple(suffix_list)):
        return True
    else:
        return False

def add_features(dataframe):
    # Initiate columns for prefix and suffix
    dataframe['hasPrefix'] = ""
    dataframe['hasSuffix'] = ""

    for i in range(len(data)):
        current_token = dataframe.loc[i]['word']

        # Save prefix and suffix in dataframe
        dataframe.loc[i, 'hasPrefix'] = contains_prefix(current_token)  # CHECK IF WORKS
        dataframe.loc[i, 'hasSuffix'] = contains_suffix(current_token)  # CHECK IF WORKS


with open(infile_path, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip().split()

        # If not an empty line create a row (dict) from the column information and add to the data list.
        if len(line) > 0:
            sent_id = f"{line[0][:2]}{line[0][-2:]}_{line[1]}"
            word = line[3]
            data.append({"sent_id": sent_id,
                         "filename": line[0],
                         "sentence_nr": line[1],
                         "token_nr": line[2],
                         "word": line[3],
                         "label": line[4]})
            # Add word to a dictionary that belongs to this sentence or start a new dict
            if sent_id in current_sentence_dict:
                current_sentence_dict[sent_id].append(word)
            else:
                current_sentence_dict[sent_id] = [word]

        # Empty line, empty the current_sentence list.
        else:
            current_sentence = []


df_data = pd.DataFrame(data)
print("Created dataframe")
extract_and_add_sentence_features(current_sentence_dict, df_data)
add_features(df_data)

# Export data with features
df_data.to_csv(outfile_path, encoding="utf-8")

print("number of sentences", len(current_sentence_dict))
print(df_data)
print(df_data.loc[:, "POS"])
print(df_data.loc[:, "lemma"])

