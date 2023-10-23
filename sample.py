from transformers import BertTokenizer, BertModel
import numpy as np
import torch

def normalize(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    norm = np.linalg.norm(embeddings)
    normalized_embeddings = embeddings / norm
    dense_embedding_list = normalized_embeddings.tolist()
    return dense_embedding_list

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Define the query and documents
query = "animal"
documents = ["dog", "cycle", "cat", "bike", "lion", "whale", "human"]
# query = "B.E."
# documents = ["Bachelor Of Engineering", "Bachelor of Science", "Bachelor of Computer Application", "Bachelor of Arts"]

query_embeddings = normalize(query, model, tokenizer)
ranked_documents = {}

for doc in documents:
    normalized = normalize(doc, model, tokenizer)
    # Calculate dot products between query and documents
    dot_products = np.dot(query_embeddings, normalized)
    ranked_documents[doc] = dot_products

# Sort the dictionary by values (dot product scores) in descending order
sorted_documents = dict(sorted(ranked_documents.items(), key=lambda x: x[1], reverse=True))

# Print the ranked documents
for key, value in sorted_documents.items():
    print(key, " - ", value)
    print(" ")
