from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load the dataset
data = pd.read_csv('dataset2.csv')

# Preprocess the texts in the dataset for similarity calculations
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'])

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    
    if result[0] == 1:
        cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)[0]
        plagiarism_sources = []
        
        threshold = 0.2
        for i, similarity in enumerate(cosine_similarities):
            if similarity > threshold:
                plagiarism_percentage = similarity * 100
                source_title = data['source_text'].iloc[i]
                plagiarism_sources.append((source_title, plagiarism_percentage))
        
        plagiarism_sources.sort(key=lambda x: x[1], reverse=True)
        detection_result = "Plagiarism Detected"
    else:
        plagiarism_sources = []
        detection_result = "No Plagiarism Detected"
    
    return detection_result, plagiarism_sources

def extract_text_from_file(file):
    text = ""
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    
    # Check if a file was uploaded
    files = request.files.getlist("files[]")
    if files:
        for file in files:
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                file_text = extract_text_from_file(file)
                input_text += "\n" + file_text
    
    detection_result, plagiarism_sources = detect(input_text)
    
    return render_template(
        'index.html', 
        result=detection_result, 
        plagiarism_sources=plagiarism_sources
    )

if __name__ == "__main__":
    app.run(debug=True)
