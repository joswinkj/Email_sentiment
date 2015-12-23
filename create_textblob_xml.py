import pandas as pd
import re,nltk

file = open('en-sentiment_backup.xml','r')
xml_text = file.read()
file.close()

neg = pd.read_csv('/home/madan/Desktop/works/pipecandy/email_analysis/mycodes/negative_words.txt',sep = '\t',encoding='ISO-8859-1')
pos = pd.read_csv('/home/madan/Desktop/works/pipecandy/email_analysis/mycodes/positive_words.txt',sep='\t',encoding='ISO-8859-1')

neg['polarity_scaled'] = -1*(1-(neg.polarity-neg.polarity.min())/(neg.polarity.max()-neg.polarity.min()))
pos['polarity_scaled'] = (pos.polarity-pos.polarity.min())/(pos.polarity.max()-pos.polarity.min())

#re.search('<word form="20th"',xml_text)

all = pd.concat([pos,neg])
file = open('new_sent.xml','a')
for ind in range(all.shape[0]):
    try:
        wrd,wrd_score = all.iloc[ind,0],all.iloc[ind,2]
        #<word form="13th"
        test_sent = '<word form="'+wrd+'"'
        if not re.search(test_sent,xml_text):
            wrd_pos_tag = nltk.pos_tag(nltk.word_tokenize(wrd))[0][1]
            write_sent = '<word form="'+str(wrd)+'" cornetto_synset_id="testid" wordnet_id="testid" pos="'+str(wrd_pos_tag)+'" sense="testsense" polarity="'+str(wrd_score)+'" subjectivity="0.5" intensity="1.0" confidence="0.9" />\n'
            file.write(write_sent)
    except:
        continue
file.close()
