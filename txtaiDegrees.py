from flask import Flask, request, jsonify
from txtai import Embeddings
import psycopg2

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
    cursor.execute("SELECT * FROM cities")  # Replace with your actual query
    data = [{"id": row[0], "name": row[1] } for row in cursor.fetchall()]
    connection.close()
    return data

# Index the data from PostgreSQL
data = fetch_data_from_postgres()
embeddings_model.index([ item["name"]  for item in data])

@app.route('/search', methods=['POST'])
def search():
    query = request.args.get('query')

    if not query:
        return jsonify([])

    # Perform a similarity search with txtai
    results = embeddings_model.search(query, limit=30)

    # Extract the top results with names and IDs as an array of objects
    top_results = [{"id": data[i[0]]["id"], "name": data[i[0]]["name"] } for i in results]

    return jsonify(top_results)

if __name__ == '__main__':
    app.run(debug=True)
