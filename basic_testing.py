import basic_analysis
text = """What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid."""
postagger = basic_analysis.POSTagger()
splitter = basic_analysis.Splitter()

splitted_sentences = splitter.split(text)
pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

dicttagger = basic_analysis.DictionaryTagger(basic_analysis.gen_pol_dict_frm_twofiles('data_files/positive_words.csv','data_files/negative_words.csv',','))
dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences,basic_analysis.default_evaluator,'newtag','newscore')

rejtagger = basic_analysis.DictionaryTagger(basic_analysis.gen_rej_dict())
rej_tagged_sentences = rejtagger.tag(pos_tagged_sentences,basic_analysis.evaluator_rej,'rejtag','rejscore')
rej_dict_tagged_sentences = rejtagger.tag(dict_tagged_sentences,basic_analysis.evaluator_rej,'rejtag','rejscore')
