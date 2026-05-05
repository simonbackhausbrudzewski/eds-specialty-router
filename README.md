# EDS Specialty Router

This is a starter solution for Lab 3: **Embeddings + intro agentic engineering**.

## Project idea

**Research question:** Can embeddings classify patient-experience text from people with Ehlers-Danlos syndromes (EDS) into likely healthcare specialty areas?

**Problem domain:** Multidisciplinary EDS care, patient-experience organization, and care-navigation support.

**Challenge:** EDS can involve many body systems. Patient descriptions may mention joint instability, dizziness, gastrointestinal problems, headaches, nerve pain, family history, or difficulty coordinating referrals. A simple keyword system is brittle because the same patient story can use many different words. Embeddings help because they represent the semantic meaning of the whole snippet.

This project is educational. It does **not** provide diagnosis, triage, emergency advice, or automatic referrals. The output should be read as a specialty category to discuss with a qualified clinician.

## Labels

- `musculoskeletal_rehab`: joint instability, subluxations/dislocations, bracing, occupational therapy, physiotherapy, strengthening, mobility aids.
- `cardiology_autonomic`: dizziness, fainting, palpitations, orthostatic symptoms, suspected POTS/dysautonomia.
- `gastroenterology`: reflux, nausea, abdominal pain, motility concerns, bowel symptoms, eating difficulty.
- `pain_neurology`: headaches, migraine-like symptoms, nerve pain, numbness/tingling, widespread pain, pain-clinic questions.
- `genetics_primary_care`: suspected EDS evaluation, family history, subtype questions, documentation, referral coordination, whole-care planning.

## Embedding model

Default model: `intfloat/multilingual-e5-small`.

This is a multilingual sentence embedding model. Each text snippet is embedded into a dense vector, then a classifier is trained on top of those vectors.

## Classifier

The model is:

- embeddings from `intfloat/multilingual-e5-small`
- `StandardScaler`
- `LogisticRegression(class_weight="balanced")`

## Files

- `data/eds_specialty_dataset.csv`: custom synthetic dataset.
- `train.py`: trains and evaluates the classifier.
- `app.py`: Gradio demo for a Hugging Face Space.
- `model/eds_specialty_classifier.joblib`: generated after training.
- `model/metrics.json`: generated after training.

## Run locally

```bash
pip install -r requirements.txt
python train.py
python app.py
```

The local demo uses port `7873` by default:

```text
http://127.0.0.1:7873
```

## Hugging Face deliverables

Create three Hugging Face repositories:

1. **Dataset repo**
   - Upload `data/eds_specialty_dataset.csv`.
   - Suggested name: `eds-specialty-router-dataset`.

2. **Model repo**
   - Run `python train.py`.
   - Upload `model/eds_specialty_classifier.joblib`, `model/metrics.json`, `train.py`, `requirements.txt`, and this README.
   - Suggested name: `eds-specialty-router-model`.

3. **Space repo**
   - Create a Gradio Space.
   - Upload `app.py`, `requirements.txt`, and the `model/` folder after training.
   - Suggested name: `eds-specialty-router-demo`.

## Evaluation note

The current dataset is intentionally small so the pipeline is easy to inspect. For a stronger assignment, expand it to around 200-500 examples. Add more Swedish examples, more realistic forum-style posts, more examples with overlapping symptoms, and examples where multiple specialties could plausibly be involved.

## Source basis

The label set is based on a multidisciplinary EDS framing. Sources consulted include:

- The Ehlers-Danlos Society patient/professional materials on physiotherapy, gastrointestinal involvement, neurological/spinal manifestations, and pain management.
- NHS-style public information on EDS management and referral to relevant professionals.
- Clinical literature describing multidisciplinary EDS clinics and care involving rheumatology/genetics, cardiology, psychology, physiotherapy/occupational therapy, and other subspecialties.

## Connection to information retrieval

This is connected to information retrieval because classification can improve how patient-experience collections are organized before search. EDS stories, forum posts, symptom notes, or appointment-preparation notes could be indexed with specialty-area labels so that users or clinicians can retrieve more relevant material.
