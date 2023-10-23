import transformers
import faiss
import numpy as np
import scipy.spatial.distance as distance

# Load the pre-trained transformer model
model = transformers.BertModel.from_pretrained("bert-base-uncased")

# Vectorize the strings that you want to search
def vectorize_string(string):
    tokens = model.tokenize(string)
    embeddings = model(tokens)[0]
    return embeddings.mean(axis=0)

# Index the vectorized strings using the faiss nearest neighbor search algorithm
def index_strings(strings):
    vectors = [vectorize_string(string) for string in strings]
    index = faiss.IndexFlatL2(vectors[0].shape[0])
    index.add(np.array(vectors))
    return index

# Perform a nearest neighbor search using the faiss index to find the most similar strings to the search query
def search_strings(query, index):
    query_vector = vectorize_string(query)
    distances, indices = index.search(query_vector, k=10)
    return indices

# Rank the results and return them to the user
def rank_results(indices, strings, query_vector):
    results = []
    for index in indices:
        result = {
            "string": strings[index],
            "score": 1 - distance.cosine(query_vector, vectors[index])
        }
        results.append(result)

    results.sort(key=lambda result: result["score"], reverse=True)
    return results

# Example usage:
strings = ["this is a string", "this is another string"]
index = index_strings(strings)
query = "a string"
query_vector = vectorize_string(query)
results = search_strings(query, index)
ranked_results = rank_results(results, strings, query_vector)

for result in ranked_results:
    print(result["string"], result["score"])
