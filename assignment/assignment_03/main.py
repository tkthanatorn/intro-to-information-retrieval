import argparse
import pandas as pd
import numpy as np
import re
import string
from ordered_set import OrderedSet
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

import nltk

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def get_and_clean_data() -> pd.DataFrame:
    data = pd.read_csv("./data/software_development_usa.csv")
    description = data["job_description"]
    cleaned_description = description.apply(
        lambda s: s.translate(str.maketrans("", "", string.punctuation + "\xa0"))
    )
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(
            str.maketrans(string.whitespace, " " * len(string.whitespace), "")
        )
    )
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description


def create_stem_cache(cleaned_description: pd.DataFrame):
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache


def create_custom_processor(stop_dict: dict, stem_cache: dict):
    def custom_processor(s: str):
        ps = PorterStemmer()
        s = re.sub(r"[^A-Za-z]", " ", s)
        s = re.sub(r"\s+", " ", s)
        s = word_tokenize(s)
        s = list(OrderedSet(s) - stop_dict)
        s = [word for word in s if len(word) > 2]
        s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
        s = " ".join(s)
        return s

    return custom_processor


def tfidf_vectorize(
    texts: list[str],
    cleaned_description: pd.DataFrame,
    stop_dict: dict,
    stem_cache: dict,
):
    my_custom_preprocessor = create_custom_processor(stop_dict, stem_cache)
    vectorizer = TfidfVectorizer(
        preprocessor=my_custom_preprocessor, use_idf=True, ngram_range=(1, 2)
    )
    vectorizer.fit(cleaned_description)

    base = vectorizer.transform(cleaned_description)
    base_df = pd.DataFrame(base.toarray(), columns=vectorizer.get_feature_names_out())

    query = vectorizer.transform(texts)
    query_df = pd.DataFrame(query.toarray(), columns=vectorizer.get_feature_names_out())

    dot_df = base_df.dot(query_df.transpose())
    search_df = pd.concat([dot_df, cleaned_description], axis=1)
    search_df = search_df.set_axis(["score", "description"], axis=1)
    search_df = search_df.sort_values(ascending=False, by="score")
    return search_df.iloc[:5]


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


def bm25_vectorize(
    texts: str,
    cleaned_description: pd.DataFrame,
    stop_dict: dict,
    stem_cache: dict,
):
    my_custom_preprocessor = create_custom_processor(stop_dict, stem_cache)
    vectorizer = TfidfVectorizer(
        preprocessor=my_custom_preprocessor, use_idf=True, ngram_range=(1, 2)
    )
    bm25 = BM25(vectorizer)
    bm25.fit(cleaned_description)
    score = bm25.transform(texts)
    search_df = pd.DataFrame({"score": score, "description": cleaned_description})
    search_df = search_df.sort_values(by="score", ascending=False)
    return search_df.iloc[:5]


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-type",
        type=str,
        choices=["bm25", "other_type"],
        help="An type of scoring model (tfidf/bm25)",
        default="tfidf",
    )

    parser.add_argument(
        "-search", type=str, help="A query you want to search in dataset"
    )

    args = parser.parse_args()
    if args.search is None or args.search == "":
        print("search argument is required.")
        return

    cleaned_description = get_and_clean_data()[:5000]
    stem_cache = create_stem_cache(cleaned_description)
    stop_dict = set(stopwords.words("english"))

    if args.type == "tfidf":
        print("[start]: tfidf model")
        result = tfidf_vectorize(
            [args.search], cleaned_description, stop_dict, stem_cache
        )
        print("[done]: tfidf model")
        print(result.head())
    elif args.type == "bm25":
        print("[start]: bm25 model")
        result = bm25_vectorize(args.search, cleaned_description, stop_dict, stem_cache)
        print("[done]: bm25 model")
        print(result.head())
    else:
        print("require type only tfidf/bm25")


if __name__ == "__main__":
    main()
