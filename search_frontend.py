from flask import Flask, request, jsonify
import pandas as pd
from collections import Counter, OrderedDict, defaultdict
import re
import nltk
import pickle
import numpy as np
from nltk.stem.porter import *
from tqdm import tqdm
import operator
from itertools import islice,count
from contextlib import closing
from functools import reduce
from operator import add
import json
from io import StringIO
from pathlib import Path
from operator import itemgetter
import sklearn
import math
import hashlib
import sys
import itertools
from itertools import islice, count, groupby
import os
from itertools import chain
import time
from sklearn import preprocessing
import itertools
from nltk.corpus import stopwords
from time import time
from google.cloud import storage
from tqdm import tqdm
import operator
from contextlib import closing
import json
from io import StringIO
import gcsfs
import builtins
nltk.download('stopwords')




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

bucket_name = 'oded_318963386'
BLOCK_SIZE = 1999998
NUM_BUCKETS = 124
TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name, prefix):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket.
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self.upload_to_gcp()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def upload_to_gcp(self):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        self._f.close()
        file_name = self._f.name
        blob = self.bucket.blob(self.prefix + "/" + f"{file_name}")
        blob.upload_from_filename(file_name)



class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes, prefix):
        b = []
        client = storage.Client()
        bucket = client.get_bucket('oded_318963386')
        for f_name, offset in locs:
            blob = bucket.get_blob(prefix + f'/{f_name}')
            if blob == None:
                continue
            pl_bin = blob.download_as_bytes()
            pl_to_read = pl_bin[offset: builtins.min(offset + n_bytes, BLOCK_SIZE)]
            n_read = builtins.min(n_bytes, BLOCK_SIZE - offset)
            b.append(pl_to_read)
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        self.doc_to_norm = {}
        self.DL_for_index = {}
        self.len_DL_for_index = 0

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs[0], self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name, prefix):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name, prefix)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp()
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name, prefix)
        return bucket_id

    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name, prefix):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(prefix + f"/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")

def sum_dict_values(dict_values):
    return reduce(add, dict_values.values())

class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, DL, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(self.DL)
        self.AVGDL = sum_dict_values(self.DL) / self.N
        self.words = list(self.index.term_total.keys())

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, prefix, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        d = {}
        for key in queries:
            Q = queries[key]
            self.idf = self.calc_idf(Q)
            for term in np.unique(Q):
                if term in self.words:
                    list_of_doc = read_posting_list(self.index, term, prefix)
                    for doc_id, freq in list_of_doc:
                        d[doc_id] = d.get(doc_id, 0) + self._score(term, doc_id, freq)
        return d

    def _score(self, term, doc_id, freq):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[str(doc_id)]
        numerator = self.idf[term] * freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
        score += (numerator / denominator)
        return score


class tf_idf_from_index:
    """
    Best Match tf-idf.
    ----------

    index: inverted index
    """

    def __init__(self, index):
        self.index = index
        self.words = list(self.index.term_total.keys())
        self.doc_to_norm = self.index.doc_to_norm
        self.DL_for_index = self.index.DL_for_index
        self.len_DL_for_index = self.index.len_DL_for_index


# regex that finds: website, dates, time, number, precent, words and html patterns
RE_WORD = re.compile(r"\b((?:www\.)?\w+\.com)\b|\b((?:0?[1-9]|[12][0-9]|3[01])\/(?:0[1-9]|1[0-2])(?:\/\d{4})?)|((?:January|March|April|May|June|July|August|September|October|November|December|Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(?:0?[1-9]|[12][0-9]|3[01])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9]|3[01])\s(?:Jan(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:(?:[12][0-9]{3})|\d{3}))|((?:February|Feb)\s(?:0?[1-9]|[12][0-9])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9])\s(?:Feb)\s(?:[12][0-9]{3}))|(\b(?:1[012]|0?[1-9])\.[0-5][0-9](?:AM|PM))\b|(\b(?:1[012]|0?[1-9])(?:\:)?[0-5][0-9](?:\s?)(?:a\.m(?:\.)?|p\.m(?:\.)?|am|pm))|(\b(?:2[0-3]|[01]?[0-9])\:(?:[0-5]?[0-9])(?:\:(?:[0-5]?[0-9]))?\b)|(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,\d{3})*(?:\.\d+)?)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+|\d)|(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,?\d{3})*(?:\.\d+)?\%)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+)|\b([A-Za-z]\.[A-Za-z]\.[A-Za-z])\b|\b([A-Za-z]\.[A-Za-z]\.(?![\w]))|(?<![\-\w])(\w+(?:\w*\-?)*\'?\w*)(?!\w*\%)|<(“[^”]*”|'[^’]*’|[^'”>])*>", re.UNICODE)

RE_WORD_5_Func = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()


# tokenizer for 5 functions:

def tokenizer_5_func(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD_5_Func.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


# tokenize assignment 4
def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords and do stemming.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


def token2bucket_id(token, i):
  return int(_hash(token),16) % NUM_BUCKETS + i * NUM_BUCKETS


def generate_query_tfidf_vector(query_to_search, index, len_DL_for_index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    len_DL_for_index: number of documents in the index

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    vocab_size = len(np.unique(query_to_search))
    Q = np.zeros((vocab_size))

    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log(len_DL_for_index / (df + epsilon), 10)  # smoothing

            try:
                ind = list(np.unique(query_to_search)).index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q



def get_candidate_documents_and_scores(query_to_search, index, words, DL_for_index, len_DL_for_index, prefix):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    DL_for_index: dictionary of document: length(document)

    len_DL_for_index: number of documents in the index

    prefix: the type of the index - string

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = read_posting_list(index, term, prefix)
            normlized_tfidf = [(doc_id, (freq / DL_for_index[doc_id]) * math.log( len_DL_for_index / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def get_top_n(sim_dict, N=100):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(int(doc_id), np.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def get_topN_score_for_queries(queries_to_search, index, doc_to_norm, DL_for_index, len_DL_for_index, words, prefix, N=100):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.

    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    DL_for_index: dictionary of document: length(document)

    len_DL_for_index: number of documents in the index

    words: list of words in the index

    prefix: the type of the index - string

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    # YOUR CODE HERE
    d = {}

    for key in queries_to_search:
        query = queries_to_search[key]
        unique_query = list(np.unique(query))
        Q = generate_query_tfidf_vector(query, index, len_DL_for_index)
        D = get_candidate_documents_and_scores(query, index, words, DL_for_index, len_DL_for_index, prefix)
        for doc_id, term in D.keys():
            d[doc_id] = d.get(doc_id, 0) + D[(doc_id, term)] * Q[unique_query.index(term)]

        q_length = np.linalg.norm(Q)
        for doc_id in d.keys():
            d[doc_id] = d[doc_id] / (q_length * doc_to_norm[doc_id])

    return {1: get_top_n(d, N)}


def get_candidate_documents_binary(query_to_search, index, words, prefix):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    prefix: the type of the index - string

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = read_posting_list(index, term, prefix)
            relevant_docs = [doc_id for doc_id, freq in list_of_doc]

            for doc_id in relevant_docs:
                candidates[(doc_id, term)] = 1

    return candidates


def generate_query_binary(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    vocab_size = len(np.unique(query_to_search))
    Q = np.zeros((vocab_size))

    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            try:
                ind = list(np.unique(query_to_search)).index(token)
                Q[ind] = 1
            except:
                pass
    return Q



def get_binary_doc_title(queries_to_search, index, words, prefix, N=40):
    """
       Generate a dictionary that gathers for every query its topN score.

       Parameters:
       -----------
       queries_to_search: a dictionary of queries as follows:
                                                           key: query_id
                                                           value: list of tokens.
       index:           inverted index loaded from the corresponding files.

       words: list of words in the index

       prefix: the type of the index - string

       N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

       Returns:
       -----------
       return: a dictionary of queries and topN pairs as follows:
                                                           key: query_id
                                                           value: list of pairs in the following format:(doc_id, score).
       """

    d = {}

    for key in queries_to_search:
        query = queries_to_search[key]
        unique_query = list(np.unique(query))
        Q = generate_query_binary(query, index)
        D = get_candidate_documents_binary(query, index, words, prefix)
        for doc_id, term in D.keys():
            d[doc_id] = d.get(doc_id, 0) + D[(doc_id, term)] * Q[unique_query.index(term)]

        q_length = np.linalg.norm(Q)
        doc_to_norm = index.doc_to_norm
        for doc_id in d.keys():
            d[doc_id] = d[doc_id] / (q_length * doc_to_norm[doc_id])

    return {1: get_top_n(d, N)}




def merge_results(list_of_scores, weights_list, N=100):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """

    # YOUR CODE HERE
    def top_n__without_round(sim_dict, N=3):
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]

    def aggregation_same_doc_id(final_dict,
                                N):  # get dictinary and N, the funcation returns update dictinary in list of pairs (doc_id, score) in length by N
        for index_query in final_dict:
            docs_id = {x[0] for x in final_dict[index_query]}  # all the doc_id
            update_tuples = []
            for i in docs_id:
                score = 0
                for x in final_dict[index_query]:
                    if x[0] == i:
                        score += x[1]
                update_tuples.append((i, score))
            # update_tuples = [(i,np.sum(x[1] for x in final_dict[index_query] if x[0] == i)) for i in docs_id]
            update_tuples_dict = dict((x, y) for x, y in update_tuples)

            final_dict[index_query] = top_n__without_round(update_tuples_dict, N)
        return final_dict

    final_dict, j = {}, 0

    for inverted_scores in list_of_scores:
        for query_index in inverted_scores:
            for i, tup in enumerate(inverted_scores[query_index]):  # i - element index, tup = (doc_id, score) - body
                old_score = inverted_scores[query_index][i][1]
                if query_index in final_dict:
                    final_dict[query_index].append((tup[0], old_score * weights_list[j]))
                else:
                    final_dict[query_index] = ([(tup[0], old_score * weights_list[j])])
        j += 1

    return aggregation_same_doc_id(final_dict, N)



def normalize_min_max(page_ranks):
    """
     Normalize the page ranks scores between 0 to 1 by  MinMaxScaler

     Parameters:
     -----------

     page_ranks: list of (doc_id ,page_rank_score)

     Returns:
     -----------
     return: list of (doc_id, normalized_page_rank_score)
     """
    # Extract the page ranks into a separate list
    ranks = [[rank] for _, rank in page_ranks]

    # Create a MinMaxScaler object and fit it to the page ranks
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(ranks)

    # Transform the page ranks using the scaler
    normalized_ranks = scaler.transform(ranks)
    normalized_ranks = [x[0] for x in normalized_ranks]
    # Zip the doc_ids and normalized ranks together into a list of tuples
    return list(zip([doc_id for doc_id, _ in page_ranks], normalized_ranks))

def weighted_average(list1, list2, list3, weight1, weight2, weight3):
    """
     Calculate the weighted average of 3 lists: index_list, page_view list and page_rank list

     Parameters:
     -----------

     list: a list of (doc_id, normalized_score)

     weight: weight for corresponding list

     Returns:
     -----------
     return: list of (doc_id, weighted_score)
     """
    # Create a dictionary mapping doc_ids to their weighted scores
    scores = {}
    for doc_id, score in list1:
        scores[doc_id] = score * weight1
    for doc_id, score in list2:
        if doc_id in scores:
            scores[doc_id] += score * weight2
        else:
            scores[doc_id] = score * weight2
    for doc_id, score in list3:
        if doc_id in scores:
            scores[doc_id] += score * weight3
        else:
            scores[doc_id] = score * weight3

    # Sort the scores in descending order and return as a list of tuples
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def read_for_search_frontend(bucket_name, name):
    """
       Read the desired file from the given bucket

       Parameters:
       -----------

       bucket_name: the bucket name the contains the file - string

       name: name of the file (string)

       Returns:
       -----------
       return: file
       """
    GCSFS = gcsfs.GCSFileSystem()
    with GCSFS.open(f"gs://oded_318963386/{name}", 'rb') as f:
        return pickle.load(f)

def read_posting_list(inverted, w, prefix):
    """
       Read the postings list of a word in an inverted index

       Parameters:
       -----------

       inverted: inverted index loaded from the corresponding files.

       w: word (string)

       prefix: the type of the index - string

       Returns:
       -----------
       return: list of (doc_id, tf)
       """
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, prefix)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))

        return posting_list


titles_dict = read_for_search_frontend(bucket_name, 'titles_dict.pkl')

DL = read_for_search_frontend(bucket_name, 'DL.pkl')
bm25_body = read_for_search_frontend(bucket_name, 'bm25_body.pkl')
bm25_title = read_for_search_frontend(bucket_name, 'bm25_title.pkl')
bm25_anchor = read_for_search_frontend(bucket_name, 'bm25_anchor.pkl')


# read inverted indexes for the 5 functions which have different tokenize function

tf_idf_body = read_for_search_frontend(bucket_name, 'tf_idf_body.pkl')
tf_idf_title = read_for_search_frontend(bucket_name, 'tf_idf_title.pkl')
tf_idf_anchor = read_for_search_frontend(bucket_name, 'tf_idf_anchor.pkl')

page_views_dict = read_for_search_frontend(bucket_name, 'page_views.pkl')
page_rank_dict = read_for_search_frontend(bucket_name, 'pr.pkl')



@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    N = 50
    query_title_anchor = {1: tokenize(query)}
    query_body = {1: tokenizer_5_func(query)}

    list_body = bm25_body.search(query_body, 'body')
    list_title = bm25_title.search(query_title_anchor, 'title')

    ranked_dict_body = {1: get_top_n(list_body, N)}
    ranked_dict_title = {1: get_top_n(list_title, N)}


    # get list of (doc_id, score)
    w1, w2, w3 = 0.1, 0.9, 0
    ranked_list = merge_results([ranked_dict_title, ranked_dict_body], [w1, w2], N)[1]
    relevant_docs = [doc_id for doc_id, score in ranked_list]
    page_ranks = [(doc_id, page_rank_dict.get(doc_id, 0)) for doc_id in relevant_docs]
    page_views = [(doc_id, page_views_dict.get(doc_id, 0)) for doc_id in relevant_docs]

    ranked_list_normalized = normalize_min_max(ranked_list)
    ranked_list_pr_normalized = normalize_min_max(page_ranks)
    ranked_list_pv_normalized = normalize_min_max(page_views)

    w_indexes, w_pr, w_pv = 0.6, 0.2, 0.2
    ranked_list = weighted_average(ranked_list_normalized, ranked_list_pr_normalized, ranked_list_pv_normalized, w_indexes, w_pr, w_pv)

    # transform the (doc_id, score) to (doc_id, title) in the list
    ranked_list = list(map(lambda x: (int(x[0]), titles_dict[x[0]]), ranked_list))[:40]
    res = ranked_list
    # END SOLUTION
    return jsonify(res)



@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    N = 40
    query = {1: tokenizer_5_func(query)}
    prefix = 'body_5_func'
    # get list of (doc_id, score)
    words = tf_idf_body.words
    ranked_list = get_topN_score_for_queries(query, tf_idf_body.index, tf_idf_body.doc_to_norm,  tf_idf_body.DL_for_index,  tf_idf_body.len_DL_for_index, words, prefix, N)[1]

    # transform the (doc_id, score) to (doc_id, title) in the list
    ranked_list = list(map(lambda x: (x[0], titles_dict[x[0]]), ranked_list))
    res = ranked_list

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    prefix = 'title_5_func'
    query = {1: tokenizer_5_func(query)}
    words = tf_idf_title.words
    ranked_list = get_binary_doc_title(query, tf_idf_title.index, words, prefix)[1]
    res = list(map(lambda x: (x[0], titles_dict[x[0]]), ranked_list))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    prefix = 'anchor_5_func'
    query = {1: tokenizer_5_func(query)}
    words = tf_idf_anchor.words
    ranked_list = get_binary_doc_title(query, tf_idf_anchor.index, words, prefix)[1]
    res = list(map(lambda x: (x[0], titles_dict[x[0]]), ranked_list))

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    res = [page_rank_dict.get(doc_id, 0) for doc_id in wiki_ids]

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    res = [page_views_dict.get(doc_id, 0) for doc_id in wiki_ids]

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
