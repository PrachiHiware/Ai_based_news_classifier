from flask import Flask, render_template, request, redirect, url_for
from bs4 import BeautifulSoup
import requests
import sqlite3
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Used for session management
db_path = "classification_history.db"

# Create a SQLite database connection
def create_connection():
    return sqlite3.connect(db_path)

# Initialize the database if it doesn't exist
def initialize_database():
    conn = create_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS classification_history (
                id INTEGER PRIMARY KEY,
                url TEXT,
                predicted_category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

initialize_database()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess the CSV data
import pandas as pd

data = pd.read_csv('BBC News Train.csv')
X = data['Text']
y = data['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a TF-IDF vectorizer and a Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Make predictions
def classify_article(article_text):
    predicted_category = pipeline.predict([article_text])
    return predicted_category[0]


# Route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        
        # Scrape article content
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = soup.get_text()

        # Classify article
        predicted_category = classify_article(article_text)

        # Store request in database
        conn = create_connection()
        with conn:
            conn.execute("INSERT INTO classification_history (url, predicted_category) VALUES (?, ?)",
                         (url, predicted_category))

        return render_template('index.html', predicted_category=predicted_category)

    return render_template('index.html', predicted_category=None)

# Route for the history page
@app.route('/history')
def history():
    conn = create_connection()
    with conn:
        rows = conn.execute("SELECT url, predicted_category, timestamp FROM classification_history ORDER BY id DESC").fetchall()
    
    #print(rows)  # Debug statement
        return render_template('history.html', history=rows)


if __name__ == '__main__':
    app.run(debug=True)

app.config['SECRET_KEY'] = os.urandom(24)
