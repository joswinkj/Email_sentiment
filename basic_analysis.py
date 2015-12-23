#from http://fjavieralba.com/basic-sentiment-analysis-with-python.html
import nltk
import pandas as pd
import re
from copy import deepcopy
import pdb

import phrases_lists

class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class Tagger(object):
    def __init__(self):
        pass
    def basic_tag(self,sentences):
        """ """
        tag = [[({'orig_string':word}, {},{}) for word in sent] for sent in sentences]
        return tag

    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        additionally sentiment score is added
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[[{'orig_string':word}, {'pos_tag':postag},{}] for (word, postag) in sentence] for sentence in pos]
        #above format: word_dict, tag_dict,score_dict
        return pos


class DictionaryTagger(object):
    def __init__(self, polarity_dic):
        ''' dictionary is the generated word dictionary'''
        self.dictionary = polarity_dic
        self.max_key_size = 0
        for key in self.dictionary:
            self.max_key_size = max(self.max_key_size,len(nltk.word_tokenize(key)))
    def tag(self, postagged_sentences_orig,tag_function,new_tag_name,new_score_name,preprocess_function = None, process_on='orig_string'):
        postagged_sentences = deepcopy(postagged_sentences_orig)
        # pdb.set_trace()
        return [self.tag_sentence(sentence,tag_function,new_tag_name,new_score_name,preprocess_function, process_on) for sentence in postagged_sentences]
    def tag_sentence(self, sentence, tag_function,new_tag_name, new_score_name ,preprocess_function = None ,process_on = 'orig_string'):
        """
        sentence : the output of the tag function of the form:
            [word_dict, tag_dict,score_dict]
        process_on : to process on what string (default is 'orig_string', which is the original string, if we are processing\
            any other type of string(eg:stemmed word), first add that to the sentence form in the word_dict with a name, then\
            pass that name to this function.
            Eg: If the dict is built after preprocessing like stemming, update the dictionary with the processed word ,
            and give a new name as key like 'stem_word', and pass that name as process_on argument ('stem_word')
        tag_function: should be a function to which all the arguments available will be passed. So it need to have a \
            **kwargs option. should return a tag name and score (score can return None if no score present)
        preprocess_function : a function which takes tagged list as input and output processed tagged_dict
            eg: stopword removal: take list, if word is stopword, remove it from the list
                stemming : take list, add a new name ('stem_word') and add the stemmed word . return the updated list

        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        if preprocess_function is not None:
            # pdb.set_trace()
            sentence = preprocess_function(sentence)
        # print(process_on)
        # print(sentence)
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        word_keys = sentence[0][0].keys()
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                # print(sentence[i:j])
                literal = ' '.join([word[0][process_on] for word in sentence[i:j]]).lower()
                # print(literal)
                # literal_orig = ' '.join([word[0]['orig_string'] for word in sentence[i:j]]).lower()
                # expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                # literal = expression_form
                if literal in self.dictionary:
                    # print('literal',literal)
                    orig_i,orig_j = i,j
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    sent_tag,sent_score = tag_function(literal,self.dictionary,is_single_token=is_single_token,\
                                                            original_position=original_position,sentence=sentence)
                    token_dict = sentence[original_position][1] if is_single_token else {'pos_tag':'New Expression'}
                    token_dict[new_tag_name] = sent_tag
                    score_dict = sentence[original_position][2] if is_single_token else {}
                    score_dict[new_score_name] = sent_score
                    #new word will be adding the individual words
                    word_dict = {}
                    for key in word_keys:
                        word_dict[key] = ' '.join([word[0][key] for word in sentence[orig_i:orig_j]])
                    new_token = [word_dict,token_dict,score_dict]
                    tag_sentence.append(new_token)
                    tagged = True
                    # print('new_token',new_token)
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

class RegexMatching(object):
    def __init__(self):
        self.rejection_regex = phrases_lists.stop_mail_exact_match_regex
    def match_sent_text_rejection(self,sentences):
        ''' '''
        if re.search(self.rejection_regex,sentences):
            return 1
        else:
            return 0

class UtilMethods(object):
    @staticmethod
    def print_tagsent(tag_sentence):
        ''' '''
        for sent in tag_sentence:
            orig_string = ''
            for word in sent:
                orig_string += word[0]['orig_string']+' '
            print('\n'+orig_string)
            print(sent)

    @staticmethod
    def default_evaluator(literal,dictionary,**kwargs):
        sent_score = dictionary[literal]
        sent_tag = 'Positive' if sent_score>0 else 'Negative'
        sent_tag = 'Neutral' if sent_score==0 else sent_tag
        return (sent_tag,sent_score)

    @staticmethod
    def gen_pol_dict_frm_twofiles(pos_path,neg_path,separator):
        neg = pd.read_csv(neg_path,sep = separator,encoding='ISO-8859-1')
        pos = pd.read_csv(pos_path,sep=separator,encoding='ISO-8859-1')
        word_scores = pd.concat([pos,neg])
        tmp = word_scores.set_index('word').to_dict()
        pol_dict = tmp['polarity']
        return pol_dict

    @staticmethod
    def gen_rej_dict():
        ''' rejection dict from phrases_lists.py file'''
        from phrases_lists import stop_contact_phrases
        out_dict = {}
        for phr in stop_contact_phrases:
            out_dict[phr] = -99
        return out_dict

    @staticmethod
    def evaluator_rej(literal,dictionary,**kwargs):
        sent_tag = 'Very bad reply'
        sent_score = None
        return (sent_tag,sent_score)

    @staticmethod
    def analyze_dict_rej(tagged_dict,tag):
        reject = 0
        for sent in tagged_dict:
            for word in sent:
                if tag in word[1]:
                    reject = 1
                    break
        return reject

    @staticmethod
    def prepro_stopword_removal(tag_sent):
        '''removing stopwords from the parsed list. here orig_string is the key '''
        # stop = nltk.corpus.stopwords.words('english')
        from phrases_lists import stopwords
        stop = stopwords
        tag_sent_new = []
        for word in tag_sent:
            if word[0]['orig_string'].lower() not in stop:
                tag_sent_new.append(word)
        return tag_sent_new

    @staticmethod
    def prepro_stemming(tag_sent):
        ''' stemming the words from the parsed list'''
        porter = nltk.stem.porter.PorterStemmer()
        for word in tag_sent:
            word[0]['stem_string'] = porter.stem(word[0]['orig_string'])
        return tag_sent