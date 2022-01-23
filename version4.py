import nltk
from nltk import pos_tag
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
pd.set_option('display.max_columns', None)


# Only run these two lines the first time
#nltk.download("wordnet")
#nltk.download('omw-1.4')

# Initiate pipeline
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('spacytextblob')

# File locations
infile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT.txt"
outfile_path = infile_path.replace(".txt", "-out.txt")

# Initiate
lemmatizer = WordNetLemmatizer()
current_sentence = []
sentence_dict = dict()
data = []

# Expression lists (partly taken from (Jan 2022): www.ef.com/wwen/english-resources/english-idioms/ & www.englishpractice.com/learning/expressions-6/ & www.espressoenglish.net/15-spoken-english-expressions-with-the-word-no/)
bi_expressions = ["no contest", "no dice", "no kidding", "no offence", "no way", "no wonder"]
tri_expressions = ["no big deal", "no hard feelings", "no harm done", "no laughing matter", "by no means", "in no time", "no matter what", "cut no ice", "cuts no ice", "no stone unturned", "not rocket science", "not brain surgery", "no hard feelings"]
tetra_expressions = ["no pain no gain", "leave no stone unturned"]
penta_expressions = ["no pain , no gain", "not my cup of tea", "not your cup of tea", "not his cup of tea", "not her cup of tea", "not their cup of tea", "not our cup of tea"]


def get_antonyms(word):
    """
    Checks if the word has antonyms and if so, returns them
    :param word: word as a string

    :returns the antonyms of the word as a list
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return antonyms


def contains_prefix(word):
    """
    Checks if a word contains a prefix
    :param word: word as a string

    :returns a boolean indicating whether the word contains a prefix
    """
    prefix_list = ['anti', 'in', 'un', 'im', 'il', 'dis', 'ir', 'an', 'non', 'a']
    if word.lower().startswith(tuple(prefix_list)) and len(word) > 2:
        return True
    else:
        return False


def get_prefix(word):
    """
    Get the prefix of a word
    :param word: word as a string

    :returns the prefix of the input word
    """
    prefix_list = ['anti', 'in', 'un', 'im', 'il', 'dis', 'ir', 'an', 'non', 'a']
    if word.lower().startswith(tuple(prefix_list)):
        return next(filter(word.lower().startswith, prefix_list))


def remove_prefix(word, prefix):
    """
    Given the word and the prefix, prefix is removed from the word and returned.
    :param word: word as a string
    :param prefix: prefix of the word as a string

    :returns the word without its prefix as a string
    """
    return word[len(prefix):]


def contains_suffix(word):
    """
    Checks if a word contains a suffix
    :param word: word as a string

    :returns a boolean indicating whether the word contains a suffix
    """
    suffix = 'less'
    if suffix in word and len(word) > 4:
        return True
    else:
        return False


def replace_suffix(word):
    """
    Given the word, the suffix 'less' is replaced by 'ful', and the newly formed word is returned.
    :param word: word as a string

    :returns the word as a string after 'less' has been replaced by 'ful'
    """
    return word.replace("less", "ful")


def pos_category(pos):
    """
    Simplifies POS-tags.
    :param pos: POS tag as a string

    :returns a simplified version of the POS tag as a string
    """
    adjectives = ['JJ', 'JJR', 'JJS']
    verbs = ['VB', 'VBD', 'VBG', 'VNB', 'VBP', 'VBZ']
    nouns = ['NN', 'NNP', 'NNPS', 'NNS']
    pronouns = ['PRP', 'PRP$']
    adverbs = ['RB', 'RBR', 'RBS']
    if pos in verbs:
        return 'VERB'
    elif pos in nouns:
        return 'NOUN'
    elif pos in adjectives:
        return 'ADJ'
    elif pos in pronouns:
        return 'PRON'
    elif pos in adverbs:
        return 'ADV'
    else:
        return 'OTHER'


def add_sentiment(df, row_index, sentence, n_gram_size=5):
    """

    :param df: the data as a pandas dataframe
    :param row_index: integer indicating the row index of the sentence in the dataframe
    :param sentence: sentence as a list of tokens
    :param n_gram_size: number of words to be in each ngram as an integer

    :returns
    """
    # n_gram_size = 5
    file_name_to_check = df.iloc[row_index]['filename']
    max_row_temp = df.loc[
        (df['sentence_nr'] == df.iloc[row_index]['sentence_nr']) & (df['filename'] == file_name_to_check)]

    max_index = max(max_row_temp.index)

    n_gram_sentiment_text = df['word'][row_index: min(row_index + (n_gram_size + 1), (max_index + 1))].to_list()
    last_word = df['word'][min(row_index + (n_gram_size + 1), (len(sentence) - 1))]
    k = ('[%s]' % ', '.join(map(str, n_gram_sentiment_text)))

    doc = nlp(k)
    sentiment = doc._.polarity
    return sentiment, last_word, n_gram_sentiment_text


def add_surrounding_token(data_frame):
    """
    Extracts the previous and next POS tag and lemma of the token and adds them to the dataframe
    :param data_frame: pandas dataframe of the data
    """
    previous_token = ['-']
    next_token = []
    for previous, next in zip(data_frame['POS'], data_frame['POS'][1:]):
        previous_token.append(previous)
        next_token.append(next)
    next_token.append('-')

    previous_lemma = ['-']
    next_lemma = []
    for previous, next in zip(data_frame['lemma'], data_frame['lemma'][1:]):
        previous_lemma.append(previous)
        next_lemma.append(next)
    next_lemma.append('-')

    data_frame['previous_pos'] = previous_token
    data_frame['next_pos'] = next_token
    data_frame['previous_lemma'] = previous_lemma
    data_frame['next_lemma'] = next_lemma

def get_ngrams(sentence, n, left=True):
    """
    Gets the left or right (as indicated by input) ngrams of given sentence and n
    :param sentence: sentence as a list of tokens
    :param n: number of words to be in each ngram as an integer
    :param left: boolean to indicate if the ngrams should be left or right padded

    :returns the ngrams
    """
    n_grams = ngrams(sentence, n, pad_left=left, pad_right=(not left), left_pad_symbol=" ", right_pad_symbol=" ")
    return n_grams

def check_for_expressions(sentence, n):
    """
    Checks for one n if any of the ngrams of sentence are an expression.
    :param sentence: sentence as a list of tokens
    :param n: number of words to be in each ngram as an integer

    :returns the number of expressions in the sentence and a list of booleans for each token in sentence to indicate if they are part of an expression
    """
    exp_count = 0
    n_grams = get_ngrams(sentence, n, left = True)
    n_grams = list(n_grams)
    is_expression = [False]*len(sentence)

    for i in range(len(n_grams)):
        n_gram = " ".join(n_grams[i])
        if (n_gram in bi_expressions) or (n_gram in tri_expressions) or (n_gram in tetra_expressions) or (n_gram in penta_expressions):
            exp_count += 1
            is_expression[i] = True
            for j in range(n):
                is_expression[i-j] = True

    return exp_count, is_expression

def check_all_ngram_exp(sentence, ns):
    """
    Checks whether the sentence contains any expression using ngrams of the sizes from ns
    :param sentence: sentence as a list of tokens
    :param ns: list of integers to indicate the number of words to be in the ngrams

    :returns the number of expressions in the sentence and a list of booleans for each token in sentence to indicate if they are part of any length n expression
    """
    exp_count = 0
    is_expression_sent = []
    final_expression_sent = []
    for n in ns:
        nr_exps, is_expression = check_for_expressions(sentence, n)
        is_expression_sent.append(is_expression)
        exp_count += nr_exps

    for i in range(len(sentence)):
        final_expression_sent.append(any([is_expression_sent[0][i], is_expression_sent[1][i], is_expression_sent[2][i], is_expression_sent[3][i]]))

    return exp_count, final_expression_sent


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
    :param dataframe: pandas dataframe of the data
    """
    dataframe["POS"] = ""
    dataframe["lemma"] = ""
    sentence_features = dict()
    exp_count = 0

    for sid, sentence in sentences.items():
        print("SID:", sid)

        # POS-tag the sentence
        pos_tagged_sent = pos_tag(sentence)

        exp_count_sent, final_expression_sent = check_all_ngram_exp(sentence, [2, 3, 4, 5])
        exp_count += exp_count_sent

        # Add sentence based features to a dictionary
        sentence_features[sid] = {"POS": pos_tagged_sent,
                                  "isExpression": final_expression_sent
                                  }

        # Add sentence based features to dataframe
        for token_index in range(len(sentence)):
            indices = dataframe.index[(dataframe["sent_id"] == sid) & (dataframe["token_nr"] == str(token_index))].to_list()
            for row_index in indices:
                sentiment, last_word,  n_gram = add_sentiment(dataframe, row_index, sentence)
                dataframe.loc[row_index, "POS"] = sentence_features[sid]["POS"][token_index][1] #[1] for the pos tag in the (token, pos) tuple
                word_pos = sentence_features[sid]["POS"][token_index][1]

                # Using NLTK Wordnet for lemmatisation
                current_token = sentence_features[sid]["POS"][token_index][0]
                pos = get_wordnet_pos(sentence_features[sid]["POS"][token_index][1])
                lemma = lemmatizer.lemmatize(current_token, pos)
                dataframe.loc[row_index, "lemma"] = lemma.lower()
                dataframe.loc[row_index, "pos_category"] = pos_category(word_pos)
                dataframe.loc[row_index, "sentiment"] = sentiment
                dataframe.loc[row_index, "isExpression"] = sentence_features[sid]["isExpression"][token_index]

    print("Number of expressions found:", exp_count)
    
def add_features(dataframe):
    """
    Extracts features and adds the features to the dataframe
    :param dataframe: pandas dataframe of the data
    """
    # Initiate columns for prefix and suffix
    dataframe['hasPrefix'] = ""
    dataframe['hasSuffix'] = ""

    for i in range(len(data)):
        current_token = dataframe.loc[i]['word']

        # Save prefix and suffix in dataframe
        dataframe.loc[i, 'hasPrefix'] = contains_prefix(current_token)  # CHECK IF WORKS
        dataframe.loc[i, 'hasSuffix'] = contains_suffix(current_token)  # CHECK IF WORKS
        if contains_prefix(current_token):
            prefix = get_prefix(current_token)
            word_without_prefix = remove_prefix(current_token, prefix)
            antonyms = get_antonyms(word_without_prefix)
            if current_token in antonyms:
                dataframe.loc[i, 'isAntonym'] = True
            else:
                dataframe.loc[i, 'isAntonym'] = False
        elif contains_suffix(current_token):
            word_with_replaced_suffix = replace_suffix(current_token)
            antonyms = get_antonyms(word_with_replaced_suffix)
            if current_token in antonyms:
                dataframe.loc[i, 'isAntonym'] = True
            else:
                dataframe.loc[i, 'isAntonym'] = False
        else:
            dataframe.loc[i, 'isAntonym'] = False



"""
START OF CODE
"""

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
                         "word": word,
                         "label": line[4]})
            # Add word to a dictionary that belongs to this sentence or start a new dict item.
            if sent_id in sentence_dict:
                sentence_dict[sent_id].append(word.lower())
            else:
                sentence_dict[sent_id] = [word.lower()]

        # Empty line, empty the current_sentence list.
        else:
            current_sentence = []

df_data = pd.DataFrame(data)
print("Created dataframe")
extract_and_add_sentence_features(sentence_dict, df_data)
add_features(df_data)
add_surrounding_token(df_data)

# Export data with features
df_data.to_csv(outfile_path, encoding="utf-8", index=False)

print("number of sentences:", len(sentence_dict))
print(df_data)
print(df_data.loc[df_data["isExpression"] == True])