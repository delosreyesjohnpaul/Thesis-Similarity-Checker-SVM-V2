from flask import Flask, render_template, request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load the dataset that the model was trained on for similarity calculation
# Assuming you have a dataset stored in 'dataset.csv' that has a column with plagiarized texts
import pandas as pd
data = pd.read_csv('dataset.csv')
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'])

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    # Prediction (Plagiarized or not)
    result = model.predict(vectorized_text)
    
    # If plagiarized is detected
    if result[0] == 1:
        # Cosine similarity to find the closest match
        cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)
        max_similarity = np.max(cosine_similarities)  # Get the highest similarity score
        plagiarism_percentage = max_similarity * 100  # Convert to percentage
        detection_result = "Plagiarism Detected"
    else:
        plagiarism_percentage = 0  # No plagiarism
        detection_result = "No Plagiarism Detected"
    
    return detection_result, plagiarism_percentage


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result, plagiarism_percentage = detect(input_text)
    return render_template('index.html', result=detection_result, percentage=plagiarism_percentage)

if __name__ == "__main__":
    app.run(debug=True)
