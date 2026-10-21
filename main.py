import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "fake_news_detection.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

FEATURES = [
    "source_credibility",
    "author_credibility",
    "clickbait_score",
    "sentiment_score",
    "article_length",
    "num_quotes",
    "num_images",
    "num_links",
    "social_shares",
    "verified_source"
]

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nLabel distribution:\n", df["label"].value_counts())

df = df.dropna(subset=FEATURES + ["label"])

X = df[FEATURES]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

scores = {}

for name, model in models.items():
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="f1_macro"
    )
    scores[name] = cv_scores.mean()
    print(f"{name} CV Macro-F1: {scores[name]:.4f}")

best_name = max(scores, key=scores.get)
best_model = models[best_name]

best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

print("\nBest model:", best_name)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro-F1:", round(f1_score(y_test, predictions, average="macro"), 4))
print("\nClassification report:\n")
print(classification_report(
    y_test, predictions, target_names=["Real", "Fake"]
))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "features": FEATURES
}, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)