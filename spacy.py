import spacy
from flask import Flask, request, jsonify


nlp = spacy.load("en_core_web_md")

# Load and process your data
db_params = {
    "host": "localhost:5432",
    "database": "db_name",
    "user": "postgres",
    "password": "Password"
}

# Function to fetch data from PostgreSQL
def fetch_data_from_postgres():
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()
    cursor.execute("SELECT name, id FROM industry")  # Replace with your actual query
    data = [{"id": row[1], "name": row[0]} for row in cursor.fetchall()]
    connection.close()
    return data

# Index the data from PostgreSQL
data = fetch_data_from_postgres()
processed_data = [nlp(item) for item in data]


app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    query = request.args.get('query')

    if not query:
        return jsonify([])

    # Process the query
    query_doc = nlp(query)

    # Perform similarity search and get the top results
    results = []
    for doc in processed_data:
        similarity = doc.similarity(query_doc)
        if similarity > 0.7:  # Adjust the similarity threshold as needed
            results.append({"text": doc.text, "similarity": similarity})

    # Sort results by similarity
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    return jsonify(results)
