import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cuml.cluster import KMeans
import cupy as cp
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from embeddinganalysis import knn as knn_funcs


model_name = "meta-llama/Llama-3.2-3B"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
prompt = "she wants to run the race"
target_token = "▁run"

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    dtype=torch.float32
)


def get_contextual_vector(prompt, target_token):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    tokens_in_prompt = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    target_token_pos = tokens_in_prompt.index(target_token)


    outputs = model(
        **inputs,
        output_hidden_states=True
    )
    last_layer = outputs.hidden_states[-1]
    target_word_vector = last_layer[0, target_token_pos, :].detach().cpu().numpy()
    
    return target_word_vector


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

race_vec = get_contextual_vector("she wanted to run the race", "▁run")
program_vec = get_contextual_vector("she wanted to run the program", "▁run")

query_pos = tokens.index("▁run")
query_vec = embeddings[query_pos].reshape(1, -1)

print(race_vec == program_vec)
print(race_vec == query_vec)
print(query_vec == program_vec)

race_neighbors = knn_funcs.get_nearest_tokens_to_vector(knn, race_vec, tokens, n=50)
print("prompt:", prompt)
print("target_token:", "race")
for token, dist in race_neighbors:
    print(f"{token:<20} distance: {dist:.4f}")
print()
print()

program_neighbors = knn_funcs.get_nearest_tokens_to_vector(knn, program_vec, tokens, n=50)
print("prompt:", prompt)
print("target_token:", "program")
for token, dist in program_neighbors:
    print(f"{token:<20} distance: {dist:.4f}")
print()
print()




