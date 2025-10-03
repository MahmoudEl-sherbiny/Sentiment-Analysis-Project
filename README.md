# Sentiment Analysis Project

A simple **Sentiment Analysis** application that loads a trained model and exposes a small API to classify text as positive / negative 

---

## Project structure

```
.
├── model/
│   └── sentiment_model.pkl          # Trained sentiment analysis model (pickle)
├── notebooks/
│   └── NLP_task.ipynb         # Jupyter notebook for experiments & training
├── src/
│   ├── app.py                 # Main application (Flask/FastAPI - serves the model)
│   └── test_request.py        # Small script to test the running API
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

> Note: paths above follow your repository layout. Make sure `model/sentiment_model.pkl` exists before running the app.

---

## Prerequisites

* Python 3.8 or higher
* `pip` package manager
* (Optional but recommended) `virtualenv` or `venv` to keep dependencies isolated

---

## Setup (recommended)

**1. Clone the repo**

```bash
git clone git@github.com:MahmoudEl-sherbiny/Sentiment-Analysis-Project.git
```

**2. Create and activate a virtual environment**

On macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> If you want to work with the Jupyter notebook (`notebooks/NLP_task.ipynb`), either add `notebook` or `jupyterlab` to your `requirements.txt` or install it manually:

```bash
pip install notebook
# or
pip install jupyterlab
```

---

## How to run the application

1. Make sure the model file `model/sentiment.pkl` is present and accessible.
2. Start the app (from the project root):

```bash
cd src
python app.py
```

This should start the server (for example, `http://127.0.0.1:5000` if using Flask, or `http://127.0.0.1:8000` for FastAPI depending on your `app.py`).

**Important:** start the app *after* you install dependencies. `app.py` requires the packages listed in `requirements.txt` to be installed.

---

## Test the API

You have two options: use the provided `test_request.py` or use `curl` / HTTP client.

**Option A — run the Python test script**

Open a new terminal (keep the server running), activate the virtual environment, then:

```bash
python src/test_request.py
```

`test_request.py` should contain a small HTTP request that sends sample text to your running API and prints the prediction.

**Option B — curl example**

```bash
# Example request (adjust URL/endpoint to match your app)
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"text": "I love this product!"}'

# Example expected response (JSON)
# { "sentiment": "positive"}
```

**Sample JSON request**

```json
{
  "text": "I really love this product!"
}
```

**Sample JSON response**

```json
{
  "sentiment": "positive"
}
```

(Your exact response fields depend on how `app.py` formats predictions.)

---

## Notes about the model

* `sentiment.pkl` should contain everything your `app.py` needs to make predictions (e.g., a pipeline that includes vectorizer/tokenizer and the classifier). If it only contains the classifier weights, make sure `app.py` rebuilds/loads the same preprocessing pipeline.
* If you retrain the model in `notebooks/NLP_task.ipynb`, export and overwrite `model/sentiment.pkl` using `joblib.dump()` or `pickle`.

Example (inside notebook):

```python
import joblib
joblib.dump(my_pipeline, "../model/sentiment.pkl")
```

---

## Common issues & troubleshooting

* `FileNotFoundError: model/sentiment.pkl` — make sure the file exists and the path in `app.py` is correct.
* `ModuleNotFoundError` or `ImportError` — re-check that you installed all packages from `requirements.txt` and activated the virtualenv.
* Port already in use — change the port in `app.py` or stop the conflicting process.
* If the model expects the same preprocessing (tokenizer, vectorizer, stopword removal), ensure the pipeline used in the notebook matches what `app.py` expects.

---

## Development tips

* To retrain the model, open `notebooks/NLP_task.ipynb`, run the experiment, then export the final pipeline to `model/sentiment.pkl`.
* Use `pytest` or similar testing framework for unit tests if you expand the project.
* Add a `.gitignore` file (recommended entries below).

**Suggested `.gitignore`**

```
venv/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
model/sentiment.pkl    # if you prefer not to upload the trained model to the repo
```

---

## Example `requirements.txt` (minimal)

```
flask
gunicorn
scikit-learn
pandas
numpy
joblib
nltk
spacy
# add notebook or jupyterlab if you want to run Jupyter
```

Adjust versions as needed.

---
