# Load model directly
from transformers import BertTokenizer,BertModel,AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Tokenize the query and documents
query="animal"
documents=["dog","cycle","cat","bike","lion","whale","human"]
query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
document_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors="pt", many=True)
# Encode the query and documents
query_embeddings = model(**query_tokens).last_hidden_state.mean(dim=1)
document_embeddings = model(**document_tokens).last_hidden_state.mean(dim=1)
print(document_embeddings,"________________________--/n|",query_embeddings,"|",)
query_embeddings=query_embeddings.detach().numpy()
document_embeddings=document_embeddings.detach().numpy()
# Calculate cosine similarity between query and documents at the sentence level
similarity_scores = cosine_similarity(query_embeddings, document_embeddings)
print(similarity_scores[0],documents,"------------------------------------------/n")
# Rank documents based on similarity
ranked_documents = [(doc, score) for doc, score in zip(documents, similarity_scores[0])]
ranked_documents.sort(key=lambda x: x[1], reverse=True)

print(ranked_documents)
result_dict = {key: value for key, value in ranked_documents}
for key,value in result_dict.items():
    print(key," - ",value)
    print(" ")
print(result_dict)


