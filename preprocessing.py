import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pyarabic.araby import strip_tashkeel, strip_tatweel
# Download the required nltk resources (only required once)
nltk.download('punkt')
nltk.download('stopwords')
def preprocess_arabic_text(text):
    text = strip_tashkeel(text)
    text = strip_tatweel(text)

    additional_symbols = r'[،؟]'

    pattern = r'[' + re.escape(additional_symbols) + ']'
    text = re.sub(pattern, '', text)

    # Remove non-Arabic characters and numbers
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('arabic'))
    words = [word for word in words if word not in stop_words]

    preprocessed_text = ' '.join(words)

    return preprocessed_text
