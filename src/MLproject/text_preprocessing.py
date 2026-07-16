import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text):

    text = str(text).lower()

    words = nltk.word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    words = [
        word
        for word in words
        if word not in stop_words
        and word not in string.punctuation
    ]

    words = [ps.stem(word) for word in words]

    return " ".join(words)