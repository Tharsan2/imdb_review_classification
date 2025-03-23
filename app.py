import joblib
import string
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess input text
def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = text.lower()  # Lowercase the text
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the input field
        review = request.form['review']
        
        # Preprocess the review
        cleaned_review = preprocess_text(review)
        
        # Convert the review to a TF-IDF feature vector
        review_tfidf = tfidf.transform([cleaned_review])
        
        # Predict sentiment using the model
        prediction = model.predict(review_tfidf)
        
        # Convert the prediction to a sentiment label
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        return render_template('index.html', prediction_text=f'The sentiment of the review is: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)
