from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz
import psycopg2

app = Flask(__name__)

# Replace with your PostgreSQL database credentials
db_connection = psycopg2.connect(
 host="localhost:5432",
 database="db_name",
 user="postgres",
 password="Password"
)

@app.route('/search', methods=['POST'])
def search():
    query = request.args.get('query')

    cursor = db_connection.cursor()
    
    # Define the table and column you want to search in
    table_name = "degrees"  # Replace with your table name
    column_name = "name"  # Replace with your column name

    # Perform fuzzy matching on the specified table and column in your database
    cursor.execute(f"SELECT {column_name} FROM {table_name}")
    data = cursor.fetchall()
    print("123")
    results = []
    for item in data:
        name = item[0]
        score = fuzz.partial_ratio(query, name)
        if score >= 80:  # You can adjust the threshold as needed
            results.append({'name': name, 'score': score})

    cursor.close()
    print("456")
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True,port=6000)
