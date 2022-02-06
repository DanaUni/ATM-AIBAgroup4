from nltk.corpus import wordnet
import pandas as pd


errorfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-dev-simple.v2-errors.csv"
df_error = pd.read_csv(errorfile_path, delimiter="\t", encoding="utf-8", keep_default_na=False)

print(df_error)

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


def get_antonyms(word):
    """
    Checks if the word has antonyms and if so, returns them
    :param word: word as a string

    :returns the antonyms of the word as a list
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        print("syn:", syn)
        #print("syn definition:", syn.definition())
        #print("syn examples:", syn.examples())
        for l in syn.lemmas():
            if l.antonyms():
                print("antonyms", l.antonyms())
                antonyms.append(l.antonyms()[0].name())
    return antonyms

#get_antonyms("credible")

error_tokens = df_error["token"].values.tolist()

yes_antonym = []

for token in error_tokens:
    print("token:", token)
    prefix = get_prefix(token)
    if prefix:
        print("yes has prefix")
        word_without_prefix = remove_prefix(token, prefix)
        print("word without prefix:", word_without_prefix)
        antonyms = get_antonyms(word_without_prefix)
        if token in antonyms:
            yes_antonym.append(token)
    else:
        get_antonyms(token)
    print()


print(yes_antonym)
