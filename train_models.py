
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    results = {
        "Naive Bayes": {
            "Accuracy": accuracy_score(y_test, nb_pred),
            "Precision": precision_score(y_test, nb_pred),
            "Recall": recall_score(y_test, nb_pred),
            "F1": f1_score(y_test, nb_pred)
        },
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, lr_pred),
            "Precision": precision_score(y_test, lr_pred),
            "Recall": recall_score(y_test, lr_pred),
            "F1": f1_score(y_test, lr_pred)
        }
    }

    return results, lr
