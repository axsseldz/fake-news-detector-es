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