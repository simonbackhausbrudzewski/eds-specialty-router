from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LABELS = [
    "musculoskeletal_rehab",
    "cardiology_autonomic",
    "gastroenterology",
    "pain_neurology",
    "genetics_primary_care",
]


def load_dataset(path: str) -> pd.DataFrame:
    candidates = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8", "skiprows": 1},
        {"sep": ";", "encoding": "utf-8-sig", "skiprows": 1},
        {"sep": ";", "encoding": "latin1", "skiprows": 1},
        {"sep": ";", "encoding": "cp1252", "skiprows": 1},
    ]

    last_error: Exception | None = None
    for options in candidates:
        try:
            df = pd.read_csv(path, **options)
            if {"text", "label"}.issubset(df.columns):
                df = df.dropna(subset=["text", "label"]).copy()
                df["text"] = df["text"].astype(str).str.strip()
                df["label"] = df["label"].astype(str).str.strip()
                df = df[df["text"] != ""]
                return df
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not parse dataset at {path}") from last_error


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    prefixed = [f"passage: {text}" for text in texts]
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eds_specialty_dataset.csv")
    parser.add_argument("--out-dir", default="model")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    unknown_labels = sorted(set(df["label"]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Dataset contains unexpected labels: {unknown_labels}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    embedding_model = SentenceTransformer(args.embedding_model)
    X_train = embed_texts(embedding_model, X_train_text)
    X_test = embed_texts(embedding_model, X_test_text)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    joblib.dump(
        {
            "embedding_model": args.embedding_model,
            "classifier": classifier,
            "labels": LABELS,
        },
        out_dir / "eds_specialty_classifier.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))
    print(f"Saved model to {out_dir / 'eds_specialty_classifier.joblib'}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
