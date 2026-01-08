### similarities baseline
# 1. get query:candidates text
# 2. sim_model functions
# 3. apply & accelerate
# 4. get evaluation metrics

from processing import *
# conda activate acc
# pip install -q rank_bm25

# --- Similarity packages ---
# (Cosine)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# (BM25)
from rank_bm25 import BM25Okapi
# baseline: BM25
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
# (Seq)
from difflib import SequenceMatcher
# (BERT)
import torch

# --- Accelerator ---
import concurrent.futures


# --- Similarity ---
def sim_bm25(ridx):
    return sim_tfidf(ridx, isBM25=True)

def sim_tfidf(ridx, isBM25=False, showSignal=False):
    # query
    query_dict = getQueryDict(ridx)
    query_corpus = getCorpus(query_dict['q'])
    query_text = ' '.join(query_corpus)
    # if showSignal:
    #     print(f'query_corpus: {query_corpus}')

    # candidates
    all_candidate = getCandidatesJSONs(ridx)
    all_candidate_texts = []
    cid_list = []
    for candidate in all_candidate:
        candidate_corpus = getCorpus(candidate['ajjbqk'])
        candidate_text = ' '.join(candidate_corpus)
        all_candidate_texts.append(candidate_text)
        cid_list.append(candidate.get('cid'))
        # if showSignal:
        #     cid = candidate.get('cid')
        #     print(f'cid: {cid}')

    # print(f'isBM25:{isBM25}')
    # print(f'cid_list:{cid_list}')
    if isBM25:
        mode = 'BM25'
        bm25 = BM25Okapi(all_candidate_texts)
        similarities = bm25.get_scores(query_corpus)
        # print(f'similarities:{similarities}')

    else:  # tfidf
        mode = 'TF-IDF'
        all_texts = [query_text] + all_candidate_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # sim
        query_vector = tfidf_matrix[0:1]
        candidates_vector = tfidf_matrix[1:]
        similarities = cosine_similarity(query_vector, candidates_vector).flatten()

    # signal
    if showSignal:
        print(f'\nsim_{mode}, ridx={ridx}')
        for cid, score in results_tfidf:
            print(f"{cid}\t{score:.6f}")
        print(f'len(cid_list): {len(cid_list)}')
        print(f'len(similarities): {len(similarities)}')

    # results
    results = list(zip(cid_list, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    cid_pred = [cid for cid, score in results]
    # print(f'cid_pred:{cid_pred}')
    return cid_pred


def sim_jaccard(corpus1, corpus2):
    """ Jaccard Similarity (词汇重叠度|lexical overlap) """
    set1 = set(corpus1)
    set2 = set(corpus2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union


# with packages: sklearn, numpy
def sim_cosine(corpus1, corpus2):
    """ Cosine Similarity (词频|term frequency) """
    text1 = ' '.join(corpus1)
    text2 = ' '.join(corpus2)
    vectorizer = CountVectorizer()  # term-freq
    tf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tf_matrix[0:1], tf_matrix[1:2])
    return similarity[0][0]


def sim_seq(corpus1, corpus2):
    """ Sequence Matching """
    text1 = ' '.join(corpus1)
    text2 = ' '.join(corpus2)
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def sim_COMP(corpus1, corpus2, models=[sim_jaccard, sim_seq], weights=[0.5, 0.5], showSignal=True):
    """ comprehensive """
    COMP = 0
    res = dict()
    for model, weight in zip(models, weights):
        name = model.__name__  # (model.__name__).split('_')[1]
        score = model(corpus1, corpus2)
        res[name] = score
        COMP += score * weight
    res['COMP'] = COMP
    if showSignal:
        for key, value in res.items():
            if not key == 'COMP':
                print(f'[{key}]: {value:.4f}')
    return COMP