from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib


MODEL_PATH = Path("model/eds_specialty_classifier.joblib")


@lru_cache(maxsize=1)
def load_model():
    from sentence_transformers import SentenceTransformer

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing trained model. Run `python train.py` first, then commit the model folder."
        )
    bundle = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(bundle["embedding_model"])
    return embedder, bundle["classifier"], bundle["labels"]


def classify(text: str):
    if not text.strip():
        return "Enter a text snippet to classify.", {}

    embedder, classifier, labels = load_model()
    embedding = embedder.encode(
        [f"passage: {text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    probabilities = classifier.predict_proba(embedding)[0]
    label_order = list(classifier.classes_)
    score_map = {label: float(probabilities[i]) for i, label in enumerate(label_order)}
    prediction = str(max(score_map, key=score_map.get))

    display_scores = {label: round(score_map.get(label, 0.0), 3) for label in labels}
    return prediction, display_scores


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Text snippet",
        lines=6,
        placeholder="Paste an EDS patient-experience note, symptom description, or care-navigation question...",
    ),
    outputs=[
        gr.Label(label="Suggested specialty category"),
        gr.JSON(label="Class probabilities"),
    ],
    title="EDS Specialty Router",
    description=(
        "Embedding-based classifier for grouping EDS patient experiences by likely specialty area: "
        "musculoskeletal rehab, cardiology/autonomic, gastroenterology, pain/neurology, or "
        "genetics/primary-care coordination. Educational only; not medical advice, diagnosis, "
        "triage, or an automatic referral."
    ),
    examples=[
        ["My shoulder slips partly out when I reach overhead and I need advice on safe strengthening."],
        ["When I stand up my heart races and I feel like I might faint."],
        ["I have reflux every night and abdominal pain after small meals."],
        ["My headaches start at the base of my skull and get worse when I sit upright."],
        ["I need a clinician to coordinate referrals because each specialist looks at only one symptom."],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7873, show_error=True)
