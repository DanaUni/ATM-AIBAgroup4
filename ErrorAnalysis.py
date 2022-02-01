from nltk.corpus import wordnet

def get_antonyms(word):
    """
    Checks if the word has antonyms and if so, returns them
    :param word: word as a string

    :returns the antonyms of the word as a list
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        print("syn:", syn)
        print("syn definition:", syn.definition())
        print("syn examples:", syn.examples())
        for l in syn.lemmas():
            if l.antonyms():
                print("", l.antonyms())
                antonyms.append(l.antonyms()[0].name())
    return antonyms

get_antonyms("unbrushed")
