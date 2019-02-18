"""Run LDA models."""
import gzip
import json
import logging
import os
import os.path
import re
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from random import shuffle, seed

import numpy as np
import pandas as pd
import scipy as sp
from scipy.special import logsumexp
# from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.matutils import jensen_shannon
from sklearn.metrics import pairwise_distances


LOGGER = logging.getLogger(__name__)

OUTDIR = "gensim_lda"
"""Directory in which model output is stored."""

MIN_DF = 10
MAX_DF = 0.9


def col_dict(cols):
    """Create ordered dict from a list of keys.

    This function is used to create

    """
    return OrderedDict(((c, []) for c in cols))


def create_doc_data(corpus):
    """Create data frame of document level data."""
    data = col_dict(("doc_id", "num_tokens", "num_words"))
    for doc_id, bow in zip(corpus['documents'], corpus['corpus']):
        data['doc_id'].append(doc_id)
        data['num_tokens'].append(sum([x[1] for x in bow]))
        data['num_words'].append(len(bow))
    return pd.DataFrame(data)


def create_word_data(corpus):
    """Word-level data.

    Parameters
    ------------
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Data frame with the following columns.

        :word: ``str``. Word
        :word_id: ``int``. Integer word id used in the corpus.
        :df: ``int``. Document frequency. Number of documents ``word`` appears.
        :tf: ``int``. Term frequency. Number of times ``word`` appears in the corpus.
        :p: ``float``.  ``tf / sum(tf)```. The proportion of the ``corpus``
            tokens.

    """  # noqa
    id2word = corpus['id2word']
    # yapf: enable
    data = {
        int(k): OrderedDict((('word', id2word[k]), ('word_id', int(k)),
                             ('tf', 0), ('df', 0)))
        for k, v in id2word.items()
    }
    # yapf: disable
    for bow in corpus['corpus']:
        for word_id, count in bow:
            data[word_id]['df'] += 1
            data[word_id]['tf'] += count
    total = sum([x['tf'] for x in data.values()])
    for k, v in data.items():
        data[k]['p'] = data[k]['tf'] / total
    return pd.DataFrame.from_records(list(data.values()))


def create_topic_data(mod, corpus, topn=20):
    """Create data frame of topic-level data.

    Parameters
    ------------
    mod: :class:`gensim.models.LdaModel`
        Fitted LDA model
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.
    num_words: int
        Number of words to use in calculating topic coherence.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Data frame with the following columns.

        - ``topic``: ``int``. topic number
        - ``tokens``: ``int``. Average number of tokens in the ``corpus``
            estimated to be from that topic.
        - ``prob`: ``float``. Proportion of tokens in the ``corpus`` from that
            topic.
        - ``prior`: ``float``. Topic prior.
            topic.
        - ``coherence``: ``float`` topic coherence calculated using the top
            ``num_words`` words in the topic.


    """
    data = {i: 0 for i in range(mod.num_topics)}
    for bow in corpus['corpus']:
        for topic, tokens in mod.get_document_topics(bow):
            data[topic] += tokens
    newdata = col_dict(("topic", "tokens"))
    for topic, tokens in data.items():
        newdata['topic'].append(topic)
        newdata['tokens'].append(tokens)
    newdata = pd.DataFrame(newdata)
    newdata['prob'] = newdata['tokens'] / newdata['tokens'].sum()
    newdata['prior'] = mod.alpha
    coherences = [
        x[1] for x in mod.top_topics(corpus['corpus'], topn=topn)
    ]
    newdata['coherence'] = coherences
    return newdata


def create_doc_word_data(corpus):
    """Document-Word-level data.

    Parameters
    ------------
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Data frame with the following columns.

        :doc_id: ``str``. Document ID.
        :word: ``str``. Word
        :tf: ``int``. Term frequency. Number of times ``word`` appears in the corpus.
        :p:`float`. Proportion of tokens in the document.

    """  # noqa
    data = []
    id2word = corpus['id2word']
    for doc_id, bow in zip(corpus['documents'], corpus['corpus']):
        n_tokens = sum(x[1] for x in bow)
        for word_id, tf in bow:
            data.append({
                # yapf: disable
                'doc_id': doc_id,
                'word': id2word[word_id],
                'tf': tf,
                'p': tf / n_tokens
                # yapf: enable
            })
    return pd.DataFrame.from_records(data)


def ecdf(arr):
    """Calculate the ECDF values for all elements in a 1D array."""
    return sp.stats.rankdata(arr, method='max') / arr.size


def frex(mod, w=0.7):
    """Calculate FREX for all words in a topic model.

    See R STM package for details.

    """
    log_beta = np.log(mod.get_topics())
    log_exclusivity = log_beta - logsumexp(log_beta, axis=0)
    exclusivity_ecdf = np.apply_along_axis(ecdf, 1, log_exclusivity)
    freq_ecdf = np.apply_along_axis(ecdf, 1, log_beta)
    out = 1. / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)
    return out


def word_freq(corpus):
    """Calculate word frequencies in a ``corpus``."""
    counts = defaultdict(int)
    for doc in corpus:
        for word, count in doc:
            counts[word] += count
    total = sum(counts.values())
    nwords = max(counts.keys())
    out = np.zeros(nwords + 1)
    for k, v in counts.items():
        out[k] = v / total
    return out


def lift(mod, p):
    r"""Word lift.

    The *lift* of word is the ratio of the topic-specific probability of a word
    to marginal probability.

    Let :math:`\beta_{kw}` be the the probability of word
    :math:`w \in 1, \dots, W` is produced by topic :math:`k \in 1, \dots, K`,
    and :math:`p_w` be the empirical frequency of word :math:`w`.
    The *lift* of word :math:`w` for topic :math:`k` is:
    .. math::

        \text{lift(w, k)} = \frac{\beta_{kw}}{p_w}
    """
    return np.exp(np.log(mod.get_topics()) - np.log(p))


def relevance(mod, p_word, w=0.5):
    r"""Word Relevance.

    Let :math:`\beta_{kw}` be the the probability of word
    :math:`w \in 1, \dots, W` is produced by topic :math:`k \in 1, \dots, K`.
    The *relevance* of a word to topic :math:`k` is:
    .. math::

        \text{relevance(w, k)} = w \log \beta_{kw} + (1 - w) \log \left(\frac{\beta_{kw}}{p_w})

    where :math:`p_{w}` is the empirical frequency of word :math:`w`, and
    :math:`w \in [0, 1]` is a weight. When :math:`w = 0`, words are ranked by
    their topic-specific weight. When :math:`w = 1`, words are ranked by lift.

    See Carson Siervert and Kenneth E. Shirley. "LDAvis: A method for visualizing and interpreting topics".

    """  # noqa
    log_beta = np.log(mod.get_topics())
    return w * log_beta + (1 - w) * (log_beta - np.log(p_word))


def create_topic_word_data(mod, corpus, frex_w=0.7, relevance_w=0.6):
    """Create data frame with topic-word information.

    Parameters
    ------------
    mod: :class:`gensim.models.LdaModel`
        Fitted LDA model
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.
    frex_w: float
        Weight to use in FREX calculations.
    relevance_w: float
        Weight to use in relevance score calculations.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Data frame with the following columns.


        :word: ``str``. Word
        :word_id: ``float``. Word identifier
        :topic: ``int``. Topic number
        :prob: ``float``. Probability of the word conditional on a topic.
        :frex: ``float``. FREX score
        :lift: ``float``. Lift score
        :relevance: ``float``. Relevance score

    """
    id2word = mod.id2word
    p_word = word_freq(corpus)
    # p(term | topic) data
    term_topics = pd.DataFrame(mod.get_topics())
    term_topics.index.name = "topic"
    term_topics.reset_index(inplace=True)
    words = pd.DataFrame({
        'word_id': list(id2word.keys()),
        'word': list(id2word.values())
    }, columns=("word_id", "word"))
    # P(word | topic)
    term_topics = pd.melt(term_topics, id_vars=["topic"],
                          var_name="word_id",
                          value_name="prob")
    term_topics['word_id']=pd.to_numeric(term_topics['word_id'])
    term_topics.merge(words, left_on="word_id", right_on="word_id")
    # P(topic | word)
    # need to ensure that the array's are flattened in the right order
    eps = 1e-10
    #term_topics['frex'] = np.ravel(frex(mod, w=frex_w), order="F")
    #term_topics['lift'] = np.ravel(lift(mod, p_word+eps), order="F")
    term_topics['relevance'] = np.ravel(relevance(mod, p_word+eps, w=relevance_w), order="F")
    return term_topics


# Dump topic distances - Jensen-Shannon distance. t-Sne or other scaling can
# be used later.
def create_topic_distances(mod, metric=jensen_shannon):
    """Topic distances.

    Inter-topic distance defined as the Jensen-Shannon distance between their
    :math:`p(word | topic)` distributions.

    Parameters
    ------------
    mod: :class:`gensim.models.LdaModel`
        Fitted LDA model
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.

    Returns
    ---------
    :class:`pandas.DataFrame`
        Data frame with the following columns.

        :doc_id: ``str``. Document string.
        :word: ``str``. Word
        :topic: ``int``. Topic number


    """
    dists = pd.DataFrame(
        pairwise_distances(mod.expElogbeta, metric=metric))
    dists.index.rename("topic_1", inplace=True)
    dists.reset_index(inplace=True)
    dists = dists.melt(
        id_vars=["topic_1"], var_name="topic_2", value_name="distance")
    dists = dists[dists['topic_1'] < dists['topic_2']].copy()
    return dists.sort_values(["topic_1", "topic_2"])


def create_doc_topic_data(mod, corpus):
    """Create Document/Topic/Term data.

    Data frame with the :math:`p(topic | document, term)` for all non-miniscule
    values. This provides document specific word probabilities.

    This is returned by the `get_document_topics` method.

    Parameters
    ------------
    mod: :class:`gensim.models.LdaModel`
        Fitted LDA model
    corpus: list
        Corpus in the same input format as required by
        :class:`gensim.models.LdaModel`.

    Returns
    ---------
    (doc_topic, doc_word_topic): tuple

        Tuple of (:class:`pandas.DataFrame`, :class:`pandas.DataFrame`) objects.
        ``doc_topic`` is data frame with the following columns.

        :doc_id: ``str``. Document string.

        :topic: ``int``. Topic number
        :count: ``int``. Number of times the word appears in the document.
        :prob: ``float``. :math:`p(topic | word, document)`. This is the
            probability of a topic conditional on both the document and topic.

        ``doc_word_topic`` is a data frame with the following columns.

        :doc_id: ``str``. Document string.
        :word: ``str``. Word
        :topic: ``int``. Topic number
        :count: ``int``. Number of times the word appears in the document.
        :prob: ``float``. :math:`p(topic | word, document)`. This is the
            probability of a topic conditional on both the document and topic.

    """  # noqa
    pmin = 1e-8
    id2word = mod.id2word
    # yapf: disable
    dtw = col_dict(("doc_id", "word", "topic", "count", "prob"))
    dt = col_dict(('doc_id', "topic", "prob"))
    # while n is redundant because it could be added to the data with
    # a merge, this is more convenient
    # yapf: enable
    for doc_id, bow in zip(corpus['documents'], corpus['corpus']):
        topics = mod. \
            get_document_topics(bow, minimum_probability=pmin,
                                per_word_topics=True)
        for topic, prob in topics[0]:
            dt['doc_id'].append(doc_id)
            dt['topic'].append(topic)
            dt['prob'].append(prob)
        for (id_, word_phi), (_, total) in zip(topics[2], bow):
            for topic, n in word_phi:
                dtw['doc_id'].append(doc_id)
                dtw['word'].append(id2word[id_])
                dtw['topic'].append(topic)
                dtw['count'].append(n)
                dtw['prob'].append(n / total)
    return (pd.DataFrame(dt), pd.DataFrame(dtw))



def create_doc_term_matrix(corpus, min_words=1):
    """Return document a term matrix of the corpus.

    Parameters
    -----------
    corpus: iterable
        An iterable corpus yielding dicts with word counts
    include: iterable
        An iterable of document_ids to include in the output corpus
    max_docs: int
        Maximum number of documents. ``None`` to include all.

    Returns
    --------
    dict
        Dictionary with the following

        - ``corpus``: dict

            - ``docs``: Number of documents
            - ``words``: Number of unique words
            - ``tokens``: Number of tokens

        - ``docs``: :class:`pandas.DataFrame`

            - ``doc_id``: document id (index)
            - ``tokens``: number of tokens in the document
            - ``words``: number of unique words in the document
            - ``p``: proportion of corpus tokens in the document

        - ``words``: :class:`pandas.DataFrame`

            - ``words``: word (index)
            - ``tf``: total number of times the word appears
            - ``df``: number of documents in which the word appears
            - ``tf_idf``: TF-IDF. f_w * log2(N / N_w)
            - ``tf_kli``: TF-KLI. f_w * log2(f_{w,d} F / f_w f_d).
                See Aizawa "The Feature Quantity: An Information Theoretic
                    Perspective of Tfidf-like Measures" 2002.
            - ``ic``: Information content of the word. See textacy package.
            - ``p``: proportion of tokens in the corpus

        - ``dtm``: :class:`pandas.DataFrame`

            - ``words``: word (index)
            - ``doc_id``: document ids (index)
            - ``count``: total number of times the word appears in the document
            - ``tf_idf``: TF-IDF
            - ``tf_kli``: TF-KLI
            - ``p``: proportion of tokens in the document

    Since this returns everything in memory, it is only for small corpora.

    """
    dtm = col_dict(("doc_id", "word", "count"))
    docs = col_dict(("doc_id", "words", "tokens"))
    words = {}
    corpus_stats = defaultdict(int)
    for doc in corpus:
        bow = doc.bag_of_words
        if len(bow) < min_words:
            continue
        corpus_stats['docs'] += 1
        corpus_stats['words'] += len(bow)
        corpus_stats['tokens'] += sum(bow.values())
        # update document level data
        docs['doc_id'].append(doc.id)
        docs['words'].append(len(bow))
        docs['tokens'].append(sum(bow.values()))
        # update term level data
        for k, v in bow.items():
            if k in words:
                words[k]['df'] += 1
                words[k]['tf'] += v
            else:
                words[k] = {}
                words[k]['word'] = k
                words[k]['df'] = 1
                words[k]['tf'] = v
        # add to document term matrix
        for k, v in bow.items():
            dtm['doc_id'].append(doc.id)
            dtm['word'].append(k)
            dtm['count'].append(v)
    corpus_stats = dict(corpus_stats)
    D = corpus_stats['docs']
    T = corpus_stats['tokens']
    dtm = pd.DataFrame(dtm).set_index(['doc_id', 'word'])
    docs = pd.DataFrame(docs).set_index('doc_id')
    words = pd.DataFrame.from_records(list(words.values()), index='word')
    # doc stats
    docs['logp'] = np.log(docs['tokens']) - np.log(T)
    # add TF-IDF like stats
    dtm = dtm.join(words)
    dtm = dtm.join(docs.rename(columns={'tokens': 'doc_tokens'})['doc_tokens'])
    dtm['tf_kli'] = dtm['count'] * (np.log(dtm['count']) - np.log(dtm['tf']) -
                                    np.log(dtm['doc_tokens']) + np.log(T))
    dtm['tf_idf'] = dtm['count'] * (np.log(D + 1) - np.log(dtm['df'] + 1))
    dtm['logp'] = np.log(dtm['count']) - np.log(dtm['doc_tokens'])
    # TF-IDF like stats to
    words['tf_idf'] = words['tf'] * (np.log(1 + D / np.log(words['df'])))
    words['tf_kli'] = dtm.groupby('word')['tf_kli'].sum()
    df = (words['df'] + 1) / (D + 1)
    words['ic'] = -df * np.log(df) - (1 - df) * np.log1p(-df)
    words['logp'] = np.log(words['tf']) - np.log(T)
    return {'corpus': corpus_stats, 'docs': docs, 'words': words, 'dtm': dtm}


def create_bow_corpus(corpus, min_df=1, max_df=.1, shuffle=True):
    """Create BOW corpus."""
    bow_corpus = create_doc_term_matrix(corpus)
    words = bow_corpus['words']
    n_docs = bow_corpus['docs'].shape[0]
    vocab = words[(words['df'] >= min_df) &
                  (words['df'] / n_docs <= max_df)].copy()
    vocab.reset_index(inplace=True)
    vocab.sort_values(
        ['tf_idf', 'word'], ascending=[False, True], inplace=True)
    vocab.reset_index(inplace=True, drop=True)
    vocab.index.rename("word_id", inplace=True)
    vocab.reset_index(inplace=True)
    vocab.set_index('word', inplace=True)
    dtm = bow_corpus['dtm'].loc[:, ['count']]. \
        join(vocab['word_id'], how='inner').copy()
    out = {}
    out['corpus'] = []
    out['documents'] = []
    for group, df in dtm.groupby('doc_id'):
        bow = zip([int(x)
                   for x in df['word_id']], [int(x) for x in df['count']])
        out['corpus'].append(list(bow))
        out['documents'].append(str(group))
    out['id2word'] = dict(
        zip([int(x) for x in vocab['word_id']], [str(x) for x in vocab.index]))
    return out


def dump_corpus(corpus, outdir):
    """Dump the values returned by ``create_bow_corpus``."""
    outfile = os.path.join(outdir, "corpus.json.gz")
    LOGGER.info(f"Writing to {outfile}")
    with gzip.open(outfile, "wt") as f:
        json.dump(corpus, f)


class ShuffledCorpus:
    """A gensim compatible corpus that yields documents in a random order."""

    def __init__(self, corpus):
        """Initialize the corpus."""
        self.documents = list(corpus)

    def __iter__(self):
        """Yield documents in random order."""
        shuffle(self.documents)
        for doc in self.documents:
            yield doc


def model_to_csv(mod, corpus, modelname, outdir):
    """Dump LDAModel ``mod`` with name ``modelname`` to files in ``outdir``."""
    # Calculate topics per document
    out_doc_topics = os.path.join(outdir, f"{modelname}-doc_topics.csv.gz")
    out_dtw = os.path.join(outdir, f"{modelname}-doc_topic_words.csv.gz")
    doc_topics, doc_topic_words = create_doc_topic_data(mod, corpus)
    LOGGER.info(f"writing to {out_doc_topics}")
    doc_topics.to_csv(out_doc_topics, index=False, compression='gzip')
    LOGGER.info(f"writing to {out_dtw}")
    doc_topic_words.to_csv(out_dtw, index=False, compression='gzip')
    out_topic_dists = os.path.join(outdir, f"{modelname}-topic_dists.csv.gz")
    LOGGER.info(f"writing to {out_topic_dists}")
    create_topic_distances(mod). \
        to_csv(out_topic_dists, index=False, compression='gzip')
    out_topics = os.path.join(outdir, f"{modelname}-topics.csv.gz")
    LOGGER.info(f"writing to {out_topics}")
    create_topic_data(mod, corpus). \
        to_csv(out_topics, index=False, compression='gzip')
    out_topic_words = os.path.join(outdir, f"{modelname}-topic_words.csv.gz")
    LOGGER.info(f"writing to {out_topic_words}")
    create_topic_word_data(mod, corpus). \
        to_csv(out_topic_words, index=False, compression='gzip')


def run_model(modelname, opts, corpus, outdir):
    """Run an LDA model with specified options.

    Run an LDA model using ``LdaMulticore` with options ``opts` using
    ``corpus`` and save results to directory ``outdir``. Call the model
    ``modelname``, which will be used to prefix the result filenames.

    """
    LOGGER.info(f"Running model {modelname}")
    mod = LdaMulticore(ShuffledCorpus(corpus['corpus']),
                       **opts, id2word=corpus['id2word'])
    filename = os.path.join(outdir, modelname + ".pickle")
    LOGGER.info(f"Saving to {filename}")
    mod.save(filename)
    model_to_csv(mod, corpus, modelname, outdir)

