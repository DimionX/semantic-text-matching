import os
import pickle
from typing import List

import pandas as pd
from load_dataset import PATH_TO_DATASET
from save_model import MODEL_PATH, VECTORIZER_PATH


def load_model() -> tuple:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model file not found")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def load_data() -> pd.DataFrame:
    if not os.path.isfile(PATH_TO_DATASET):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(PATH_TO_DATASET)
    df.columns = ["dr_id", "q1", "q2", "label"]

    return pd.concat([df["q1"], df["q2"]]).unique()


def load() -> tuple:
    model, vectorizer = load_model()
    docs = load_data()

    return model, vectorizer, docs


def matching(question: str, top_n: int = 5) -> List[str]:
    model, vectorizer, docs = load()

    vectors = vectorizer.transform([question])
    pred_doc = model.kneighbors(vectors, return_distance=False)

    for i, indexes in enumerate(pred_doc):
        return [docs[index] for index in indexes if docs[index] != question][:top_n]


if __name__ == "__main__":
    q = input("Question: ")

    print(matching(q, 5))
