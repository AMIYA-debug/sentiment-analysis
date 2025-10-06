#  Review Sentiment Flask App

This small Flask app serves a web page that classifies a  review as positive (1) or negative (0) using the preprocessing pipeline from the project's notebook and the saved model/vectoriser.

Files added

- `app.py` - Flask app that loads `vectoriser.model` and `model.pkl` and exposes two routes: `/` and `/predict`.
- `templates/index.html` - input form.
- `templates/home.html` - displays prediction.
- `requirements.txt` - suggested packages to install.

Notes and assumptions

- The notebook saved `vectoriser.model` using `gensim` keyed vectors. `app.py` expects that file to be loadable with `KeyedVectors.load("vectoriser.model")`.
- The sklearn model is loaded from `model.pkl` and should accept a single vector shaped (1, n_features). The pipeline in the notebook used average word2vec vectors; `app.py` mirrors that behavior.
- NLTK resources (stopwords, wordnet) will be downloaded at runtime if missing. Ensure the environment has internet access first-run.

Running locally (Windows PowerShell):

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

If `KeyedVectors.load` fails because the vectoriser was saved with `save_word2vec_format`, replace the load call in `app.py` with `KeyedVectors.load_word2vec_format(path, binary=...)` matching the file format.
