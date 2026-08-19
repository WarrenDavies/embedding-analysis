from sklearn.preprocessing import normalize as sk_normalize

def get_nearest_tokens_to_token(
    knn,
    query_token: str,
    tokens,
    whole_word_tokens,
    embeddings,
    n: int = 10,
):
    """
    Given a token string (with or without the ▁ prefix),
    return the n nearest tokens by cosine similarity.
    """
    # Normalise the query token to include the ▁ prefix if missing
    if not query_token.startswith('▁'):
        query_token = '▁' + query_token

    if query_token not in whole_word_tokens:
        raise ValueError(f"Token '{query_token}' not found in whole_word_tokens.")

    # Get the position of the query token in our filtered list
    query_pos = tokens.index(query_token)
    query_vec = embeddings[query_pos].reshape(1, -1)

    distances, neighbor_positions = knn.kneighbors(query_vec, n_neighbors=n + 1)

    results = []
    for dist, pos in zip(distances[0], neighbor_positions[0]):
        neighbor_token = tokens[pos]
        if neighbor_token == query_token:
            continue  # skip the query token itself
        results.append((neighbor_token.lstrip('▁'), float(dist)))

    return results[:n]


def get_nearest_tokens_to_vector(
    knn,
    vector,
    tokens,
    n: int = 10,
):

    query_vec = sk_normalize(vector.reshape(1, -1))

    distances, neighbor_positions = knn.kneighbors(query_vec, n_neighbors=n)

    results = []
    for dist, pos in zip(distances[0], neighbor_positions[0]):
        neighbor_token = tokens[pos]
        results.append((neighbor_token.lstrip('▁'), float(dist)))

    return results