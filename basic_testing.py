import basic_analysis
import sys

utils = basic_analysis.UtilMethods()

def get_tagged_out(text,senttag,sentscore,rejtag,rejscore):
    tagger = basic_analysis.Tagger()
    splitter = basic_analysis.Splitter()

    splitted_sentences = splitter.split(text)
    tagged_sentences = tagger.basic_tag(splitted_sentences)

    dicttagger = basic_analysis.DictionaryTagger(utils.gen_pol_dict_frm_twofiles('data_files/positive_words.csv','data_files/negative_words.csv',','))
    dict_tagged_sentences = dicttagger.tag(tagged_sentences,utils.default_evaluator,senttag,sentscore)

    rejtagger = basic_analysis.DictionaryTagger(utils.gen_rej_dict())
    # rej_tagged_sentences = rejtagger.tag(pos_tagged_sentences,basic_analysis.evaluator_rej,'rejtag','rejscore')
    rej_dict_tagged_sentences = rejtagger.tag(dict_tagged_sentences,utils.evaluator_rej,rejtag,rejscore)
    return rej_dict_tagged_sentences

def test_code(text=None):
    if text is None:
        text = """What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid."""
    tagged_text = get_tagged_out(text,'sent_tag','sent_score','rej_tag','rej_score')
    utils.print_tagsent(tagged_text)
    reject_decision = 'Reject' if utils.analyze_dict_rej(tagged_text,'rej_tag')==1 else 'Not Reject'
    print(reject_decision)

if __name__ == '__main__':
    if len(sys.argv)>1:
        text=sys.argv[1]
        test_code(text)
    else:
        test_code()




