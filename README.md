# Semantic Text Matching

ML-model for solving the problem of searching semantically similar text documents.

The model processes a new incoming text question and returns a list of N similar questions from an existing [dataset](#3-download-dataset) of 4,567 medical-related questions.

## Model

Unsupervised learner for implementing neighbor searches: [Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn-neighbors-nearestneighbors)

Metric to use for distance computation: `cosine`


## Preprocessing

1. Tokenization: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
   * `tokenizer`: custom function with regular pattern `r"(?u)\b\w\w+\b"`
   * `token_pattern`: `None`
2. Stop words: [nltk.corpus.stopwords](https://www.nltk.org/api/nltk.corpus.html)
3. Minimum word length: `2`
4. Remove punctuation
5. Decode special symbols: `html.unescape`
6. Lemmatization: [spaCy](https://spacy.io/usage/models/) `en_core_web_sm` model

## Input

`question` - `string` input question

## Output

`questions` - `List[str]` output similar questions (from [Dataset](#3-download-dataset))

## Metric

[Top-N accuracy](https://www.baeldung.com/cs/top-n-accuracy-metrics) (accuracy@n)

`Accuracy@5` - `0.887`


## Run

### 1. Clone repo

```bash
git clone https://github.com/DimionX/semantic-text-matching.git

cd semantic-text-matching
```

### 2. Creation of virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Dev:

```bash
pip install -r requirements.dev.txt
pre-commit install
```

Production:

```bash
pip install -r requirements.txt
```


### 3. Download dataset

Dataset info - [Medical Questions Pairs](https://huggingface.co/datasets/medical_questions_pairs)

```bash
python ./src/load_dataset.py
```

### 4. Model Training

#### Load spacy models

```bash
python -m spacy download en
```

#### Model training and save

```bash
python ./src/save_model.py
```

## Jupyter Notebook

[main.ipynb](notebooks/main.ipynb)


## Run Streamlit

```bash
streamlit run ./src/stream.py
```

Streamlit app in your browser: http://127.0.0.1:8501

## Run CLI

```bash
python ./src/matcher.py
```

## Run Docker Compose

```bash
docker compose up
```

Streamlit app in your browser: http://127.0.0.1:8501