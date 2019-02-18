import logging
import gensim
from gensim import corpora
from gensim.corpora import WikiCorpus, MmCorpus
from gensim import models

from pprint import pprint
from collections import defaultdict, Counter
from gensim.parsing.preprocessing import strip_non_alphanum, strip_numeric, strip_short, strip_multiple_whitespaces

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

import lda_functions

import pyLDAvis.gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#run params
stop_cities = False
min_docs = 1
max_percent_limit = 0.1
n_topics = 15
passes = 15
n_test = 1000
test_run = False
load_previous_wiki_filtered = True
load_previous_lda_model = False
load_previous_similarity = False

#set display width
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 10)
np.set_printoptions(linewidth=desired_width)
#use R file to generate this data file first, can't read into pandas for some reason
df = pd.read_csv('./dockerize/data/candidate_answers_utf8.csv', sep=';', encoding='utf-8', engine='python')

# get relevant columns
cols_to_select = []
cols_to_select.append(list(df[['kunta','sukunimi','etunimi','puolue','ikä','sukupuoli','valittu','sitoutumaton',
                  'ehdokasnumero','Äidinkieli','Miksi.juuri.sinut.kannattaisi.valita.kunnanvaltuustoon.',
                  'Mitä.asioita.haluat.edistää.tai.ajaa.tulevalla.vaalikaudella.',
                  'Koulutus','Poliittinen.kokemus','Käytän.vaaleihin.rahaa']].columns.values))

cols_to_select.append(list(df.filter(regex="kommentti").columns.values))
cols_to_select.append(list(df.filter(regex="Vaalilupaus").columns.values))
cols_to_select = [item for sublist in cols_to_select for item in sublist]

df = df.loc[:, cols_to_select]

#replace nan with empty string
df = df.replace(np.nan, '', regex=True)
#define text cols to use
join_cols = df.loc[:,'X127.kommentti':'Vaalilupaus.3'].columns
join_cols = join_cols.append(df.loc[:,['Miksi.juuri.sinut.kannattaisi.valita.kunnanvaltuustoon.',
       'Mitä.asioita.haluat.edistää.tai.ajaa.tulevalla.vaalikaudella.']].columns)

#join the all text answers for a general statment
df['text'] = df[join_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df.loc[df.text == '                                      ','text'] = np.nan
df['text']=df['text'].str.strip()
#drop the candidates without text answers
df.dropna(subset=['text'], inplace=True)
#reset column 'index' to row numbers
df['rownum'] = np.arange(len(df))

#define documents
documents = df['text']
documents.reset_index(drop=True, inplace=True) #reset index to 0->len(documents)
# remove common words and tokenize

#testing
if test_run:
    df = df.iloc[0:n_test,:]
    documents = documents.iloc[0:n_test]
    if len(df) != len(documents):
        print("wrong lengths for df and documents!")

stopwords = pd.read_json('https://raw.githubusercontent.com/stopwords-iso/stopwords-fi/master/stopwords-fi.json')
stoplist = set(stopwords.iloc[:,0].unique())
if stop_cities:
    cities = ['espoo', 'helsinki', 'turku', 'tampere', 'jyväskylä', 'kuopio', 'oulu',
              'espoon', 'helsingin', 'turun', 'tampereen', 'jyväskylän', 'kuopion',
              'oulun', 'kouvola', 'kouvolan', 'vaasa', 'vaasan', 'lahti', 'lahden',
              'kauhava', 'kauhavan', 'salo', 'salon', 'turussa', 'helsingissä', 'espoossa',
              'joensuun', 'kotkan', 'keravan', 'hämeenlinnan', 'joensuun', 'mikkelin',
              'vantaan', 'vihdin']
    for city in cities:
        stoplist.add(city)


texts = [[word for word in strip_short(strip_multiple_whitespaces(strip_numeric(strip_non_alphanum(document.lower()))), minsize=3).split() if word not in stoplist and word!=[]]
         for document in documents]
if len(df) != len(texts):
    print("wrong lengths for df and texts!")
if len(texts) != len(documents):
    print("wrong lengths for texts and documents!")

#
# # remove words that appear only once
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
#
# texts = [[token for token in text if frequency[token] > 1]
#          for text in texts]


rows_to_drop = [n for n in np.arange(0,len(texts)) if texts[n]==[]]
for doc in sorted(rows_to_drop, reverse=True):#reverse because otherwise indices don't match anymore
    del texts[doc]
documents = documents[~documents.index.isin(rows_to_drop)]
documents.reset_index(drop=True, inplace=True)
df = df[~df.rownum.isin(rows_to_drop)]
if len(df) != len(texts):
    print("wrong lengths for df and texts vol 2!")
if len(documents) != len(texts):
    print("wrong lengths for documents and texts vol 2!")

with open("./dockerize/data/wiki_cats.pkl", 'rb') as meta_file:
    pages_to_retain = pickle.load(meta_file)

categories_to_keep = pages_to_retain.category.unique()
page_titles_in_correct_categories = pages_to_retain.page_title.str.replace("_", " ").unique()


def filter_selected_categories(xml_element, page_titles_in_correct_categories=page_titles_in_correct_categories, **kwargs):
    for elem in xml_element.iter('{http://www.mediawiki.org/xml/export-0.10/}title'):
        if elem.text in page_titles_in_correct_categories:
            return xml_element
    return None


if not load_previous_wiki_filtered:
    wiki = gensim.corpora.WikiCorpus('fiwiki-20190101-pages-articles.xml.bz2', filter_articles=filter_selected_categories)
    gensim.corpora.mmcorpus.MmCorpus.serialize(fname="./dockerize/data/orig_wiki/wiki_filtered.mm", corpus=wiki,
                                               metadata=True, progress_cnt=1000)
    wiki.dictionary.save_as_text('./dockerize/data/orig_wiki/wiki_filtered_wordids.txt.bz2')

id2word = gensim.corpora.Dictionary.load_from_text('./dockerize/data/orig_wiki/wiki_filtered_wordids.txt.bz2')
wiki_bow = gensim.corpora.MmCorpus('./dockerize/data/orig_wiki/wiki_filtered.mm')

with open("./dockerize/data/orig_wiki/wiki_filtered.mm.metadata.cpickle", 'rb') as meta_file:
    titles = pickle.load(meta_file)


print("Wiki complete!")

dictionary_elec = id2word
#dictionary_elec.filter_extremes(no_below=min_docs, no_above=max_percent_limit)
#dictionary_elec.compactify()
corpus_elec = [dictionary_elec.doc2bow(text) for text in texts]
#drop documents that did not have any words from wikipedia dictionary
documents_to_drop = [n for n in np.arange(0,len(corpus_elec)) if corpus_elec[n] == []]
print(len(texts))
print(len(documents))
print(len(corpus_elec))
print(len(df))

for doc in sorted(documents_to_drop, reverse=True):#reverse because otherwise indices don't match anymore
    print(doc)
    del texts[doc]
    del corpus_elec[doc]

[n for n in np.arange(0,len(corpus_elec)) if corpus_elec[n] == []]
[doc for doc in documents_to_drop if doc in documents.index]
documents = documents[~documents.index.isin(documents_to_drop)]
documents.reset_index(drop=True, inplace=True)

df = df[~df.rownum.isin(documents_to_drop)]
print(len(texts))
print(len(documents))
print(len(corpus_elec))
print(len(df))


print("Corpus complete!")

if load_previous_lda_model:
    lda_elec = gensim.models.LdaModel.load('./dockerize/saved_models/lda_elec.lda')
else:
    # lda_elec = models.LdaModel(wiki_bow, id2word=dictionary_elec, num_topics=n_topics,
    #                           passes=passes, update_every=0)
    lda_elec = gensim.models.LdaMulticore(corpus=wiki_bow, id2word=dictionary_elec, num_topics=n_topics,
                                          passes=passes, workers=4, random_state=123)
    if not test_run:
        lda_elec.save('./dockerize/saved_models/lda_elec.lda')

elec_lda_corpus = lda_elec[corpus_elec]
[n for n in np.arange(0,len(elec_lda_corpus)) if elec_lda_corpus[n] == []]

top_topics = [doc[0][0] if doc != [] else -1 for doc in elec_lda_corpus]
print(Counter(top_topics))

if not test_run:
    MmCorpus.serialize("./dockerize/data/elec_lda_corpus.mm",elec_lda_corpus)

print("LDA complete!")

#index = gensim.similarities.MatrixSimilarity(elec_lda_corpus, num_features=lda_elec.num_terms)

#topic_data = lda_functions.create_topic_word_data(mod=lda_elec, corpus=wiki_bow, frex_w=0.7, relevance_w=0.0)
#topic_data['word'] = topic_data.apply(lambda x: dictionary_elec.id2token[x['word_id']], axis=1)
#most_relevant_words = topic_data.groupby(by='topic')['topic','word','relevance','prob'].apply(lambda x: x.nlargest(5, columns=['relevance']))

#pprint(most_relevant_words)

#vis = pyLDAvis.gensim.prepare(lda_elec, corpus_elec, dictionary_elec)
#pyLDAvis.show(vis)




# load corpus iterator

#wiki_tfidf = tfidf[wiki_bow]
#wiki_lda = lda_elec[wiki_bow]

if test_run:
    index = gensim.similarities.Similarity(output_prefix=None, corpus=lda_elec[wiki_bow], num_features=lda_elec.num_terms)
else:
    if load_previous_similarity:
        index = gensim.similarities.Similarity.load('./wiki_lda/saved_models/similarity_elec.index')
    else:
        index = gensim.similarities.Similarity(output_prefix="shard", corpus=wiki_bow, num_features=lda_elec.num_terms)
        index.save('./wiki_lda/saved_models/similarity_elec.index')



print("indexing complete!")

sims = index[lda_elec[corpus_elec[0]]] #perform similarity match against the corpus
sims = sorted(enumerate(sims), key=lambda item: -item[1])#sort in descending order
res = pd.DataFrame({"ordered_pos":[x[0] for x in sims[0:20]],
                    "distance": [x[1] for x in sims[0:20]]})
res['title'] = [titles[x][1] for x in res['ordered_pos']]
pprint(res)


def get_closest_wiki_document(cand_text_no, similarity_index, nclosest, titles):

    # similarities = similarity_index[lda_elec[corpus_elec[cand_text_no]]] # perform similarity match against the corpus
    # similarities = sorted(enumerate(similarities), key=lambda item: -item[1])  # sort in descending order
    # result = pd.DataFrame({"ordered_pos": [x[0] for x in similarities[0:20]],
    #                        "distance": [x[1] for x in similarities[0:20]]})
    # result['title'] = [titles[x][1] for x in result['ordered_pos']]
    # return '; '.join(str(x) for x in result['title'][0:nclosest])
    return '; '.join(str(x) for x in [titles[x][1] for x, _ in sorted(enumerate(similarity_index[lda_elec[corpus_elec[cand_text_no]]]), key=lambda item: -item[1])[0:nclosest]])

print("starting nums")
#tqdm.pandas()
nums = [index[lda_elec[x]] for x in corpus_elec]
print("nums done")
nums_sorted = [sorted(enumerate(x), key=lambda item: -item[1])[0:5] for x in nums]
print("nums sorted")
wiki_titles = [[titles[x][1] for x, _ in item] for item in nums_sorted]
print("titles fetched")
titles_merged = ["; ".join(y) for y in wiki_titles]
print("wiki titles merged")

print("most common titles:")
pprint(Counter(titles_merged).most_common(10))


print("df.apply ran successfully!")

df['wiki_closest'] = "aa"
df.loc[:,'wiki_closest'] = titles_merged
print("titles set to df!")


if not test_run:
    df.to_csv('./wiki_lda/data/cand_df_filled.csv')
    df.to_pickle('./wiki_lda/data/cand_df.pkl')
else:
    df.to_csv('./wiki_lda/data/cand_df_filled_test.csv')
    df.to_pickle('./wiki_lda/data/cand_df_test.pkl')

print("script complete!")
