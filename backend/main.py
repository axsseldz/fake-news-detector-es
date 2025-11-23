import os
from pathlib import Path
import re
import pickle
import sys
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
from gensim.models import Word2Vec
from joblib import load as joblib_load
from pydantic import BaseModel
from sklearn.svm import SVC, LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WORD2VEC_V1_DIR = MODELS_DIR / "word2vec_v1"
WORD2VEC_V2_DIR = MODELS_DIR / "word2vec_v2"
BETO_DIR = MODELS_DIR / "beto_finetuned"
BETO2_DIR = MODELS_DIR / "beto2_finetuned"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Fake News Detector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsInput(BaseModel):
    text: str


class GenerationRequest(BaseModel):
    flavor: Literal["fake", "true"]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def v1_tokenizer(text: str):
    return _tokenize(text)


def v2_tokenizer(text: str):
    return _tokenize(text)


def v1_tokenize(text: str):
    return _tokenize(text)


def v2_tokenize(text: str):
    return _tokenize(text)


if "__mp_main__" in sys.modules:
    mp_main = sys.modules["__mp_main__"]
    mp_main.v1_tokenizer = v1_tokenizer  # type: ignore
    mp_main.v2_tokenizer = v2_tokenizer  # type: ignore
    mp_main.v1_tokenize = v1_tokenize  # type: ignore
    mp_main.v2_tokenize = v2_tokenize  # type: ignore


def _load_word2vec_stack(
    model_dir: Path, w2v_filename: str, tfidf_filename: str, clf_filename: str
) -> Dict[str, object]:
    w2v_path = model_dir / w2v_filename
    tfidf_path = model_dir / tfidf_filename
    clf_path = model_dir / clf_filename

    w2v_model = Word2Vec.load(str(w2v_path)) if w2v_path.exists() else None
    tfidf = joblib_load(tfidf_path) if tfidf_path.exists() else None

    try:
        clf = joblib_load(clf_path)
    except Exception:
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

    return {"w2v": w2v_model, "tfidf": tfidf, "clf": clf}


def _load_beto_stack(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def _vectorize(tokens: List[str], w2v: Word2Vec) -> np.ndarray:
    vectors = [w2v.wv[word] for word in tokens if word in w2v.wv]
    if not vectors:

        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


try:
    w2v_v1_stack = _load_word2vec_stack(
        WORD2VEC_V1_DIR, "w2v_v1.model", "tfidf_v1.pkl", "linearsvc_v1.pkl"
    )
    w2v_v2_stack = _load_word2vec_stack(
        WORD2VEC_V2_DIR, "w2v_v2_bt.model", "tfidf_v2_bt.pkl", "linearsvc_v2_bt.pkl"
    )
    beto_tokenizer, beto_model = _load_beto_stack(BETO_DIR)
    beto2_tokenizer, beto2_model = _load_beto_stack(BETO2_DIR)
except Exception as exc:  
    raise RuntimeError(f"Failed to load models: {exc}") from exc