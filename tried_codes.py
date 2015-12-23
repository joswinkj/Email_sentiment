import nltk

from phrases_lists import stop_mail_exact_match

responses = ['Yes. This is of interest. I am out of the country all through next week. Though I can have a call \
tomorrow (Saturday) at 1.30 p.m. Do let me know if this works?', 'We are not doing mobile development at this time - \
and we insource all our development given we are protective of our IP (patent pending status). We are looking for \
Javascript programmers in The Philippines - so if you can make introductions to resources here in The Philippines \
who might be looking for a Job with upside - please feel free to connect them with us :)', 'Could you send me more \
information? Yes we interesting in mobile apps, but I have to know more about your services.', 'Thanks for reaching \
out.\xc2\xa0 I would be interested in learning more.\xc2\xa0 How do you suggest we do th is?', 'Thanks for your mail,\
 but unfortunately we have no need for a different company who can assist us in ecommerce infrastructure.', 'On my way \
 back to office will check schedule and get back to you shortly', 'Thank you for contacting me, but since this is\
  something out of my responsibilities, I would suggest you to contact Mr Demetre Valindras VP IT Southeastern Europe',\
             "What's your ask?", "No thanks! Unfortunately this is not a fit for us at our scale\n\nIf you'd like to\
              buy custom watches for your team or partner, we'd be happy to work on them with you. We've collaborated\
               with Nike, the NBA, Facebook, Google, SendGrid and a bunch more. Please let me know"]

###chinking
def try_chinking(responses):
    grammar = r"""
      NP:
        {<.*>+}          # Chunk everything
        }<VBD|IN>+{      # Chink sequences of VBD and IN
      """
    cp = nltk.RegexpParser(grammar)
    for sent in responses:
        sentence = nltk.pos_tag(nltk.word_tokenize(sent.decode('utf8')))
        tmp = cp.parse(sentence)
        for i in tmp:
            if type(i) == nltk.tree.Tree:
                i.flatten()
            else:
                print(i)

##chunking
def try_chunking(responses):
    grammar = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    cp = nltk.RegexpParser(grammar)
    for sent in responses:
        sentence = nltk.pos_tag(nltk.word_tokenize(sent.decode('utf8')))
        tmp = cp.parse(sentence)
        for i in tmp:
            if type(i) == nltk.tree.Tree:
                i.flatten()
            else:
                print(i)


def try_exactmatch(responses):
    for sent in responses:
        if re.search(stop_mail_exact_match,sent):
            print(sent)