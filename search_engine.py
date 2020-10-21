import re
import numpy as np
import pandas as pd
from pymystem3 import Mystem
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def get_stop_words():
    global russian_stopwords
    with open('russian', 'r', encoding='utf-8') as file:
        text = file.read()
        russian_stopwords = text.split('\n')


def get_mystem():
    global m
    m = Mystem()


def get_queries():
    global queries_base
    queries_base = pd.read_csv('queries.csv')


def get_answers():
    global answers_base
    answers_base = pd.read_csv('answers.csv')


def get_model():
    global model
    model_file = 'model_files/araneum_none_fasttextcbow_300_5_2018.model'
    model = KeyedVectors.load(model_file)


def preprocess(text):
    text = text.lower()
    text = re.sub('[^а-яё]', ' ', text)
    text = [token for token in text.split() if token not in russian_stopwords]
    text = ' '.join(text)
    return text


def preprocess_lemmas(text):
    text = text.lower()
    text = re.sub('[^а-яё]', ' ', text)
    lemmas = m.lemmatize(text)
    text = [lemma for lemma in lemmas if lemma not in russian_stopwords]
    text = " ".join(text)
    return text


def normalize_vec(v):
    return v / np.sqrt(np.sum(v ** 2))


def query_base_indexing_bm25():
    global tokens_corpus, bm25
    tokens_corpus = queries_base.clean_text.tolist()
    tokenized_corpus = [doc.split() for doc in tokens_corpus]
    bm25 = BM25Okapi(tokenized_corpus)


def query_base_indexing_tfidf():
    global tfidf_matrix, vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_corpus = queries_base.lemmatized_text.tolist()
    tfidf = vectorizer.fit_transform(tfidf_corpus)
    tfidf_matrix = tfidf.toarray()


def query_base_indexing_w2v_basic():
    global w2v_basic_matrix
    vectors = []
    for query in tokens_corpus:
        tokens = query.split()
        tokens_vectors = np.zeros((len(tokens), model.vector_size))
        vec = np.zeros((model.vector_size,))
        for idx, token in enumerate(tokens):
            if token in model:
                tokens_vectors[idx] = model[token]
        if tokens_vectors.shape[0] is not 0:
            vec = np.mean(tokens_vectors, axis=0)
        vec = normalize_vec(vec)
        vectors.append(vec)
    w2v_basic_matrix = np.matrix(vectors)


def w2v_advanced_index_single_doc(text):
    lemmas = text.split()
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))

    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lemmas_vectors[idx] = normalize_vec(model[lemma])
    return lemmas_vectors


def query_base_indexing_w2v_advanced():
    global w2v_advanced_index
    w2v_advanced_index = []
    for query in tokens_corpus:
        query_matrix = w2v_advanced_index_single_doc(query)
        w2v_advanced_index.append(query_matrix)
    return w2v_advanced_index


def bm25_search(query):
    query = preprocess(query)
    tokenized_query = query.split()
    answer_text = bm25.get_top_n(tokenized_query, tokens_corpus, n=1)
    answer_id = queries_base[queries_base['clean_text'] == answer_text[0]].iloc[0]['answer_id']
    answer_text = answers_base[answers_base['answer_id'] == answer_id]['preview'].values[0]
    answer = f"{answer_id} {answer_text}"
    return answer


def tfidf_search(query):
    query = preprocess_lemmas(query)
    query_tdidf_vector = vectorizer.transform([query]).toarray()
    cosine_similarities = linear_kernel(query_tdidf_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    answer_doc = related_docs_indices[0]
    answer_id = queries_base['answer_id'][answer_doc]
    answer_text = answers_base[answers_base['answer_id'] == answer_id]['preview'].values[0]
    answer = f"{answer_id} {answer_text}"
    return answer


def w2v_basic_search(query):
    tokens = preprocess(query).split()
    tokens_vectors = np.zeros((len(tokens), model.vector_size))
    vec = np.zeros((model.vector_size,))
    for idx, token in enumerate(tokens):
        if token in model:
            tokens_vectors[idx] = model[token]
    if tokens_vectors.shape[0] is not 0:
        vec = np.mean(tokens_vectors, axis=0)
    vec = normalize_vec(vec)
    query_matrix = np.matrix(vec)
    cosine_similarities = linear_kernel(query_matrix, w2v_basic_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    answer_doc = related_docs_indices[0]
    answer_id = queries_base['answer_id'][answer_doc]
    answer_text = answers_base[answers_base['answer_id'] == answer_id]['preview'].values[0]
    answer = f"{answer_id} {answer_text}"
    return answer


def w2v_advanced_search(query):
    query = preprocess(query)
    query_matrix = w2v_advanced_index_single_doc(query)
    sims = []
    for doc in w2v_advanced_index:
        sim = doc.dot(query_matrix.T)
        sim = np.max(sim, axis=0)
        sims.append(sim.sum())
    answer_doc = np.argmax(sims)
    answer_id = queries_base['answer_id'][answer_doc]
    answer_text = answers_base[answers_base['answer_id'] == answer_id]['preview'].values[0]
    answer = f"{answer_id} {answer_text}"
    return answer



