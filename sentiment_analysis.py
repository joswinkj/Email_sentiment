import pandas as pd
import nltk

def gen_pol_dict():
    neg = pd.read_csv('/home/madan/Desktop/works/pipecandy/email_analysis/mycodes/negative_words.txt',sep = '\t',encoding='ISO-8859-1')
    pos = pd.read_csv('/home/madan/Desktop/works/pipecandy/email_analysis/mycodes/positive_words.txt',sep='\t',encoding='ISO-8859-1')
    word_scores = pd.concat([pos,neg])
    tmp = word_scores.set_index('word').to_dict()
    pol_dict = tmp['polarity']
    return pol_dict

def word_pol(word,pol_dict):
    if word in pol_dict.keys():
        return pol_dict[word]
    return 0

pol_dict = gen_pol_dict()
def get_polarity(content):
    words = nltk.word_tokenize(content)
    pol_scr , n_wrds = 0,0
    for wrd in words:
        wrd_scr = word_pol(wrd,pol_dict)
        if wrd_scr != 0:
            pol_scr += wrd_scr
            n_wrds += 1
    if n_wrds > 0:
        pol_scr = pol_scr/n_wrds
    return pol_scr
