import os
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from flask import Flask, request
from elasticsearch import Elasticsearch
import time
import pandas as pd
from pathlib import Path
import pickle
import os
import json
import pandas as pd
import numpy as np
import os
from scipy import sparse

import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------- WebIndexer Class-----------------------------------
class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        if not isinstance(vectorizer, TfidfVectorizer):
            raise ValueError("Vectorizer must be an instance of TfidfVectorizer")
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        # Fit the vectorizer and transform the document set
        self.vectorizer.fit(X)
        self.y = self.vectorizer.transform(X)

        # Calculate the average document length
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        # Ensure the input query is a list
        if not isinstance(q, list):
            q = [q]

        # Transform the query using the vectorizer
        q_vector = self.vectorizer.transform(q)
        assert sparse.isspmatrix_csr(q_vector)

        # Calculate BM25 scores
        len_y = self.y.sum(1).A1
        y = self.y.tocsc()[:, q_vector.indices]
        denom = y + (self.k1 * (1 - self.b + self.b * len_y / self.avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q_vector.indices] - 1
        numerator = y.multiply(np.broadcast_to(idf, y.shape)) * (self.k1 + 1)
        return (numerator / denom).sum(1).A1

class WebIndexer:
    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')).parent / 'crawled/'
        self.stored_file = './manual_indexer.pkl'
        if os.path.isfile(self.stored_file):
            with open(self.stored_file, 'rb') as f:
                cached_dict = pickle.load(f)
                self.__dict__.update(cached_dict)
        else:
            self.run_indexer()
    
    def pre_process(self, s: str):
        return s.lower()

    def run_indexer(self):
        documents = []
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        tfidf_vectorizer = TfidfVectorizer(preprocessor=self.pre_process, stop_words=stopwords.words('english'))
        self.bm25 = BM25(tfidf_vectorizer)
        self.bm25.fit(self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1))
        with open(self.stored_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

# ----------------------------- Application Class-----------------------------------
app = Flask(__name__)
app.es_client = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "abcd1234"))
app.indexer = WebIndexer()

@app.route('/search_es', methods=['GET'])
def search_es():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    results = app.es_client.search(
        index='simple',
        source_excludes=['url_lists'],
        size=100,
        query={"match": {"text": query_term}}
    )
    end = time.time()
    total_hit = results['hits']['total']['value']
    results_df = pd.DataFrame([
        [hit["_source"]['title'], hit["_source"]['url'], hit["_source"]['text'][:100], hit["_score"]]
        for hit in results['hits']['hits']
    ], columns=['title', 'url', 'text', 'score'])

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object

@app.route("/search_manual", methods=["GET"])
def search_manual():
    start = time.time()
    response_object = {"status": "success"}
    argList = request.args.to_dict(flat=False)
    query = argList["query"][0]
    score = app.indexer.bm25.transform(query)
    document = app.indexer.documents
    document["score"] = score
    document = document.sort_values(["score"], ascending=False)
    document = document[document["score"] != 0]
    results = list(document.T.to_dict().values()) 
    
    response_object["total_hit"] = len(results)
    response_object["results"] = results
    response_object["elapse"] = time.time() - start
    return response_object

@app.route('/search', methods=['GET'])
def search():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    results = app.es_client.search(index='simple', source_excludes=['url_lists'], size=100,
                                   query={"script_score": {"query": {"match": {"text": query_term}},
                                                          "script": {"source": "_score * doc['pagerank'].value"}}})
    end = time.time()
    total_hit = results['hits']['total']['value']
    results_df = pd.DataFrame([[hit["_source"]['title'], hit["_source"]['url'], hit["_source"]['text'][:100],
                                hit["_score"]] for hit in results['hits']['hits']],
                              columns=['title', 'url', 'text', 'score'])

    response_object['total_hit'] = total_hit
    response_object['results'] = results_df.to_dict('records')
    response_object['elapse'] = end - start

    return response_object



if __name__ == "__main__":
    app.run()