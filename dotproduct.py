from transformers import BertTokenizer, BertModel
import numpy as np

def normalize(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    # dense_embedding_list = embeddings.tolist()
    norm = np.linalg.norm(embeddings)
    normalized_embeddings = embeddings / norm
    dense_embedding_list = normalized_embeddings.tolist()


# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Define the query and documents
query = "animal"
documents = ["dog", "cat", "tiger", "bike", "lion", "whale", "human","car","cycle"]
# query = "B.E."
# documents = ["Bachelor Of Engineering", "Bachelor of Science", "Bachelor of Computer Application", "Bachelor of Arts"]


# Tokenize the query and documents
query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
document_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors="pt", many=True)

# Encode the query and documents
query_embeddings = model(**query_tokens).last_hidden_state.mean(dim=1)
document_embeddings = model(**document_tokens).last_hidden_state.mean(dim=1)

# Convert embeddings to NumPy arrays
query_embeddings = query_embeddings.detach().numpy()
document_embeddings = document_embeddings.detach().numpy()

# Calculate dot products between query and documents
dot_products = np.dot(query_embeddings, document_embeddings.T)

# Print dot products
print(dot_products, documents)

# Rank documents based on dot products
ranked_documents = [(doc, score) for doc, score in zip(documents, dot_products[0])]
ranked_documents.sort(key=lambda x: x[1], reverse=True)

print(ranked_documents)
result_dict = {key: value for key, value in ranked_documents}
for key, value in result_dict.items():
    print(key, " - ", value)
    print(" ")
print(result_dict)


