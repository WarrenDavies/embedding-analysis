from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from cuml.cluster import KMeans
import cupy as cp
import numpy as np
import pandas as pd

# from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from cuml.metrics.cluster import silhouette_score as cuml_silhouette
# from cuml.metrics.cluster import davies_bouldin_score as cuml_db
# from cuml.metrics.cluster import calinski_harabasz_score as cuml_ch
from cuml.cluster import KMeans
import cupy as cp
from cuml.decomposition import PCA
from cuml.manifold import UMAP




model_name = "meta-llama/Llama-3.2-3B"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    dtype=torch.float32
)
vocab = tokenizer.get_vocab()

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

pca = PCA(n_components=100)
embeddings_gpu = pca.fit_transform(embeddings_gpu)


print(embeddings_gpu.shape)
results = {
    "k": [],
    "inertia_scores": [],
    "silhouette_scores": [],
    "davies_bouldin_scores": [],
    "calinski_harabasz_scores": [],
}

k_min = 2
k_max = 500
k_step = 10
for k in range(k_min, k_max, k_step):

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings_gpu)

    labels_gpu = kmeans.labels_
    inertia = kmeans.inertia_
    labels = labels_gpu.get()
    

    silhouette     = cuml_silhouette(embeddings_gpu, labels_gpu)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)

    results["k"].append(k)
    results["inertia_scores"].append(float(inertia))
    results["silhouette_scores"].append(float(silhouette))
    results["davies_bouldin_scores"].append(float(davies_bouldin))
    results["calinski_harabasz_scores"].append(float(calinski_harabasz))

    print(f"k: {k}", end=" | ")
    print(f"Silhouette: {silhouette:.3f}", end=" | ")
    print(f"Inertia: {inertia:.3f}", end=" | ")
    print(f"davies_bouldin: {davies_bouldin:.3f}", end=" | ")
    print(f"calinski_harabasz: {calinski_harabasz:.3f}", end=" | ")
    print()


df_results = pd.DataFrame(results)
print(df_results)