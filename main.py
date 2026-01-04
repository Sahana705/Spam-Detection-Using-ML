
import pandas as pd
from src.data_preprocessing import preprocess_text
from src.feature_extraction import get_vectorizer
from src.train_models import train_and_evaluate
from src.predict import predict_message

data = pd.read_csv("data/spam_dataset.csv")
data["EmailText"] = data["EmailText"].apply(preprocess_text)

X_text = data["EmailText"]
y = data["Label"]

vectorizer = get_vectorizer()
X = vectorizer.fit_transform(X_text)

results, model = train_and_evaluate(X, y)

for name, metrics in results.items():
    print(f"\n{name} Performance")
    for k, v in metrics.items():
        print(f"{k}: {v}")

sample = "You have won a free lottery prize"
print("\nPrediction:", predict_message(model, vectorizer, sample))
