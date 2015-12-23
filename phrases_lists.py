#detecting emails with very negative sentiment, ie customers who need to be removed
import re
import nltk
#one option is to try to directly match
stop_contact_phrases = ['stop send','stop sending',"don't send","dont send",'do not send','stop contact','stop contacing',\
                        "stop mail",'stop mailing',"don't mail",'do not mail',\
                        'never send mail','never send email','never send e mail','do not email','do not contact',"don't contact",\
    'never contact',"don't spam","dont spam",'stop spam','stopping spam','send spam','sending spam','spam inbox','spamming inbox',\
                        'remove address','removing address','remove from mailing list','removal of mailing list',\
                        'remove email','removing email','remove mail','removing mail',\
                        'remove from mail','removing from mail','delete mail','deleting mail','delete email','deleting email',\
                        'delete e-mail','deleting e-mail','remove e-mail','removing e-mail','any more email',\
                        'anymore email']
stop_mail_exact_match_regex = re.compile('|'.join(stop_contact_phrases),re.IGNORECASE)
#another option is to stem words and match phrases (no need to match spamming and spam separately

# stopwords = nltk.corpus.stopwords.words('english')
stopwords = ['my','your','ur']