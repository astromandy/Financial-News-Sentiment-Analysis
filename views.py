import re
import string
from nltk.corpus import stopwords
import joblib

# Load the vectorizer and model only once (at module level)
vectorizer = joblib.load('vectorizer_trained_on_training_data.pkl')  # update with actual path
model = joblib.load('model_trained_with_that_vectorizer.pkl')        # update with actual path

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # ...add any other steps from your notebook...
    return text

def predict_view(request):
    # ...existing code...
    if request.method == 'POST':
        user_input = request.POST.get('headline')
        preprocessed_input = preprocess_text(user_input)
        # Only transform, never fit!
        X = vectorizer.transform([preprocessed_input])
        prediction = model.predict(X)
        # ...existing code...