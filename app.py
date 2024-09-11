from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocess the texts in the dataset for similarity calculations
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'])

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    # Predict whether the input is plagiarized
    result = model.predict(vectorized_text)
    
    if result[0] == 1:
        # Calculate cosine similarity to find all similar plagiarized texts
        cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)[0]
        plagiarism_sources = []
        
        # Set a threshold for relevant similarity (> 20%)
        threshold = 0.2
        for i, similarity in enumerate(cosine_similarities):
            if similarity > threshold:
                plagiarism_percentage = similarity * 100
                source_title = data['source_text'].iloc[i]
                plagiarism_sources.append((source_title, plagiarism_percentage))
        
        # Sort the sources by the highest percentage
        plagiarism_sources.sort(key=lambda x: x[1], reverse=True)
        detection_result = "Plagiarism Detected"
    else:
        plagiarism_sources = []  # No plagiarism sources
        detection_result = "No Plagiarism Detected"
    
    return detection_result, plagiarism_sources


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result, plagiarism_sources = detect(input_text)
    
    return render_template(
        'index.html', 
        result=detection_result, 
        plagiarism_sources=plagiarism_sources
    )


if __name__ == "__main__":
    app.run(debug=True)
