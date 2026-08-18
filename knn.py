import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cuml.cluster import KMeans
import cupy as cp
import numpy as np
import pandas as pd

# from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from cuml.cluster import KMeans
import cupy as cp
from cuml.decomposition import PCA
from cuml.manifold import UMAP

from embeddinganalysis import knn as knn_funcs


model_name = "meta-llama/Llama-3.2-3B"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
search_token = "London"

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    dtype=torch.float32
)

whole_word_tokens = {
    token: idx for token, idx in vocab.items()
    if token.startswith('▁') and token[1:].isalpha()
}
indices = list(whole_word_tokens.values())
tokens  = list(whole_word_tokens.keys())

embedding_matrix = model.get_input_embeddings().weight
embeddings = embedding_matrix
embeddings = embeddings[indices].detach().cpu().numpy()
embeddings = normalize(embeddings)
embeddings_gpu = cp.asarray(embeddings)


knn = NearestNeighbors(n_neighbors=10, metric='euclidean', algorithm='brute')
knn.fit(embeddings)

neighbors = knn_funcs.get_nearest_tokens_to_token(knn, search_token, tokens, whole_word_tokens, embeddings, n=50)

for token, dist in neighbors:
    print(f"{token:<20} distance: {dist:.4f}")




