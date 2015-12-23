#detecting emails with very negative sentiment, ie customers who need to be removed
import re
#one option is to try to directly match
stop_contact_phrases = ['stop send',"don't send","dont send",'stop contact',"stop mail","don't mail",'never send mail'\
    ,'never contact me',"don't spam","dont spam",'stop spam','send spam','sending spam','spam inbox','spamming my inbox',\
                        'spamming inbox','sorely dissapointed']
stop_mail_exact_match = re.compile('|'.join(stop_contact_phrases),re.IGNORECASE)
#another option is to stem words and match phrases (no need to match spamming and spam separately