from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import KMeans
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


model_name = "meta-llama/Llama-3.2-3B"

concepts = ["Self(I)", "Number(2)", "Quantity(More)", "Quality(Good)", "Negation(Not)", "Preposition(For)"]
test_tokens_by_language = {
    "english": ["I", "two", "more", "good", "not", "for"],
    "spanish": ["yo", "dos", " más", " bien", "no", "por"],
    "german": ["ich", " zwei", " mehr", " gut", " nicht", " für"],
    "french": ["je", " deux", "plus", "bon", "pas", "pour"],
    "italian": ["io", "due", " più", " bene", "non", "per"],
    "portuguese": ["eu", " dois", " mais", " bom", " não", "para"],
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    dtype=torch.float32
)
vocab = tokenizer.get_vocab()

test_data = []
for language in test_tokens_by_language:
    for i, concept in enumerate(test_tokens_by_language[language]):
        token_ids = tokenizer.encode(concept, add_special_tokens=False)
        assert len(token_ids) == 1
        test_data.append({
            "language": language,
            "word": concept,
            "concept": concepts[i],
            "token_id": token_ids[0],
        })
for test_data_ in test_data:
    print(test_data_)

token_ids = [token["token_id"] for token in test_data]

embedding_matrix = model.get_input_embeddings().weight
embeddings = embedding_matrix[token_ids]
embeddings = embeddings.detach().cpu().numpy()
embeddings = normalize(embeddings)



kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(embeddings)

labels = kmeans.labels_
print(labels)


centers = kmeans.cluster_centers_
print(centers)


for i, data_dict in enumerate(test_data):
    data_dict['cluster'] = int(kmeans.labels_[i])

print(test_data)

for item, label in zip(test_data, kmeans.labels_):
    print(
        item["language"],
        item["word"],
        label
    )

for i in range(6):
    cluster_datapoints = [ token for token in test_data if token["cluster"] == i ]
    print(f"Cluster {i}")
    print()
    for cluster_datapoint in cluster_datapoints:
        print(cluster_datapoint["word"] + ": ", cluster_datapoint["language"], cluster_datapoint["concept"])
    print()
    

df_results = pd.DataFrame(test_data)

print(kmeans.inertia_)
score = silhouette_score(
    embeddings,
    kmeans.labels_
)
print(score)
print(df_results.pivot(index="language", columns="concept", values="cluster"))