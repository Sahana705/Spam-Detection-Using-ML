
from src.data_preprocessing import preprocess_text

def predict_message(model, vectorizer, message):
    message = preprocess_text(message)
    vector = vectorizer.transform([message])
    return "Spam" if model.predict(vector)[0] == 1 else "Not Spam"
