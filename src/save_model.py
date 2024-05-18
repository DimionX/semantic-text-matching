import pickle
import platform
from os import path

import pandas as pd
from load_dataset import PATH_TO_DATASET
from sklearn.neighbors import NearestNeighbors
from vectorizer import Vectorizer

TOP_N = 10
METRIC = "cosine"
N_JOBS = -1
MODEL_PATH = path.join("models", f"model_{platform.system()}.pkl")
VECTORIZER_PATH = path.join("models", f"vectorizer_{platform.system()}.pkl")


def main():
    if not path.isfile(PATH_TO_DATASET):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(PATH_TO_DATASET)
    documents = pd.concat([df["question_1"], df["question_2"]]).unique()

    vectorizer = Vectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    model = NearestNeighbors(n_neighbors=TOP_N + 1, metric=METRIC, n_jobs=N_JOBS)
    model.fit(tfidf_matrix)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    main()
