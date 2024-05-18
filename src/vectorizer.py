from functools import lru_cache
from html import unescape
from re import findall
from string import punctuation
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer(TfidfVectorizer):
    TOKEN_PATTERN = r"(?u)\b\w\w+\b"

    def __init__(self):
        super().__init__(tokenizer=self.get_tokens, token_pattern=None)

        nltk.download("stopwords", quiet=True)
        self.STOPWORDS = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm")

    @lru_cache(maxsize=100000)
    def is_correct_word(self, word: str) -> bool:
        return (len(word) > 1) and (word not in self.STOPWORDS) and (word not in punctuation)

    def get_tokens(self, sent: str) -> List[str]:
        """
        :param sent: sentence with a question
        :return: array of tokens (with lemmatization)
        """
        sent_prep = unescape(sent).strip().lower()
        words = findall(self.TOKEN_PATTERN, sent_prep)
        sent_clean = " ".join(words)

        return [w.lemma_ for w in self.nlp(sent_clean) if self.is_correct_word(w.lemma_)]
