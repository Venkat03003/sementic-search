from flask import Flask, request, jsonify
from txtai import Embeddings
import psycopg2
import json

app = Flask(__name__)

# Initialize the Similarity model with a pre-trained model
embeddings_model = Embeddings({"path": "facebook/bart-large-cnn"})

# Define your PostgreSQL database connection parameters
db_params = {
    "host": "localhost:5432",
    "database": "stg_admin",
    "user": "postgres",
    "password": "Password"
}

# Function to fetch data from PostgreSQL
def fetch_data_from_postgres():
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()
    cursor.execute("SELECT name, id FROM cities")  # Replace with your actual query
    data = [{"id": row[1], "name": row[0]} for row in cursor.fetchall()]
    connection.close()
    return data

# Index the data from PostgreSQL
data = fetch_data_from_postgres()
# def load_data_from_json(filename):
#     with open(filename, "r") as file:
#         data = json.load(file)
#     return data

# # Load data from a JSON file
# data = load_data_from_json("industry.json")

embeddings_model.index([item["name"] for item in data])
results = embeddings_model.search("madras", limit=30)

    # Extract the top results with names and IDs as an array of objects
top_results = [{"id": data[i[0]]["id"], "name": data[i[0]]["name"]} for i in results]

print(jsonify(top_results))

@app.route('/search', methods=['POST'])
def search():
    query = request.args.get('query')

    if not query:
        return jsonify([])

    # Perform a similarity search with txtai
    results = embeddings_model.search(query, limit=30)

    # Extract the top results with names and IDs as an array of objects
    top_results = [{"id": data[i[0]]["id"], "name": data[i[0]]["name"]} for i in results]

    return jsonify(top_results)

if __name__ == '__main__':
    app.run(debug=True)
