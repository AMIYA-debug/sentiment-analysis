import os
import re
import numpy as np
import pickle

BASE_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(BASE_DIR, "vectoriser.model")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


vectoriser = None
model = None
vector_size = None


def load_resources():
    """Load the gensim KeyedVectors and the sklearn model into module globals.
    Tries multiple loaders to be robust to different saved formats.
    """
    global vectoriser, model, vector_size
    try:
        from gensim.models import KeyedVectors
    except Exception as e:
        raise

    loaded = None
    try:
        loaded = KeyedVectors.load(VECT_PATH, mmap='r')
    except Exception:
        try:
            loaded = KeyedVectors.load_word2vec_format(VECT_PATH, binary=True)
        except Exception:
            loaded = KeyedVectors.load_word2vec_format(VECT_PATH, binary=False)

    vectoriser = loaded
    vector_size = getattr(vectoriser, 'vector_size', None)

    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)


def cleaning(sentence):
    
    review = re.sub('[^a-zA-z]', ' ', str(sentence))
    review = review.lower()
    review = review.split()
    return review


def processing(sentence):
    
    words = []
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        wl = WordNetLemmatizer()
        
        try:
            stopwords.words('english')
        except Exception:
            nltk.download('stopwords')
        try:
            WordNetLemmatizer()
        except Exception:
            nltk.download('wordnet')

        for i in range(0, len(sentence)):
            if sentence[i] not in stopwords.words('english'):
                sentence[i] = wl.lemmatize(sentence[i], pos='v')
                words.append(sentence[i])
    except Exception:
        
        for tok in sentence:
            if len(tok) > 1:
                words.append(tok)

    return words


def avgw2v(doc):
    
    global vectoriser, vector_size
    if vectoriser is None:
        raise RuntimeError('vectoriser not loaded - call load_resources() first')
    vecs = [vectoriser[word] for word in doc if word in vectoriser.key_to_index]
    
    if not vector_size:
        try:
            _size = vectoriser.vector_size
        except Exception:
            _size = 300
    else:
        _size = vector_size

    if len(vecs) == 0:
        
        return np.zeros(_size, dtype=float)
    return np.mean(vecs, axis=0)


def preprocess_text(text):
    toks = cleaning(text)
    toks = processing(toks)
    return toks


def predict_review(text):

    global model
    if model is None or vectoriser is None:
        raise RuntimeError('Resources not loaded. Call load_resources() first.')
    toks = preprocess_text(text)
    vec = avgw2v(toks)
    X = vec.reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        label = int(model.predict(X)[0])
        prob = float(probs[label])
    else:
        label = int(model.predict(X)[0])
        prob = None
    return label, prob
