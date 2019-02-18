import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('./data/wiki_wordids.txt.bz2')
# load corpus iterator
mm = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm.bz2') # use this if you compressed the TFIDF output (recommended)

print(mm)

#make LDA representation of Wikipedia
# extract 100 LDA topics, using 20 full passes, no online updates
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=10)
lda.save('./saved_models/lda_model.lda')