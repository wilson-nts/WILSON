"""
a modification from gensim
"""

import logging
from gensim.utils import deprecated
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.bm25 import get_bm25_weights as _bm25_weights
from gensim.corpora import Dictionary
from math import log10 as _log10
from six.moves import xrange
import networkx as nx


INPUT_MIN_LENGTH = 10

WEIGHT_THRESHOLD = 1.e-3

logger = logging.getLogger(__name__)

def _build_graph(sequence):
    graph = nx.DiGraph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph

def _set_graph_edge_weights(graph):
    documents = list(graph.nodes())
    weights = _bm25_weights(documents)
    
    ########## create directed graph
    for i in xrange(len(documents)):
        for j in xrange(len(documents)):
            if i == j or weights[i][j] < WEIGHT_THRESHOLD:
                continue
            
            sentence_1 = documents[i]
            sentence_2 = documents[j]

            if not graph.has_edge(sentence_1, sentence_2):
                graph.add_edge(sentence_1, sentence_2, weight=weights[i][j])

def _get_doc_length(doc):
    return sum([item[1] for item in doc])

def _get_similarity(doc1, doc2, vec1, vec2):
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)

    denominator = _log10(length_1) + _log10(length_2) if length_1 > 0 and length_2 > 0 else 0

    return numerator / denominator if denominator != 0 else 0

def _build_corpus(sentences):
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    return [dictionary.doc2bow(token) for token in split_tokens]


def _get_important_sentences(sentences, corpus, important_docs):
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = {}
    for k, v in zip(hashable_corpus, sentences):
        if k not in sentences_by_corpus:
            sentences_by_corpus.setdefault(k, v)
        elif len(v.text) < len(sentences_by_corpus[k].text):
            sentences_by_corpus[k] = v
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, word_count):
    length = 0
    selected_sentences = []

    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        if abs(word_count - length - words_in_sentence) > abs(word_count - length):
            return selected_sentences

        selected_sentences.append(sentence)
        length += words_in_sentence

    return selected_sentences


def _extract_important_sentences(sentences, corpus, important_docs, word_count):
    important_sentences = _get_important_sentences(sentences, corpus, important_docs)

    return important_sentences \
        if word_count is None \
        else _get_sentences_with_word_count(important_sentences, word_count)


def _format_results(extracted_sentences, split):
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _build_hasheable_corpus(corpus):
    return [tuple(doc) for doc in corpus]


def summarize_corpus(corpus, num=1):
    hashable_corpus = _build_hasheable_corpus(corpus)

    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph)

    if len(graph.nodes()) == 0:
        logger.warning("Please add more sentences to the text. The number of reachable nodes is 0")
        return []

    pagerank_scores = nx.pagerank_numpy(graph) #_pagerank(graph)
    

    hashable_corpus.sort(key=lambda doc: (round(pagerank_scores.get(doc, 0), 12), -len(doc)), reverse=True)

    return [list(doc) for doc in hashable_corpus[:num]]


def summarize(text, num=1, word_count=None, split=False, rerank=True):
    sentences = _clean_text_by_sentences(text)

    if len(sentences) == 0:
        logger.warning("Input text is empty.")
        return [] if split else u""

    corpus = _build_corpus(sentences)

    most_important_docs = summarize_corpus(corpus, num=num if word_count is None else 1)

    if not most_important_docs:
        logger.warning("Couldn't get relevant sentences.")
        return [] if split else u""

    extracted_sentences = _extract_important_sentences(sentences, corpus, most_important_docs, word_count)

    if rerank:
        extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split)