A fully modular, production-grade academic chatbot built using FastAPI, spaCy, and scikit-learn.
Designed for answering academic queries through intent classification, entity extraction, and template-based responses.

🚀 1. Project Overview

This project implements a rule-augmented NLP system consisting of:

✔ Intent Classification

Using TF-IDF + LinearSVC (sklearn) to classify user messages into one of 27 academic intents.

✔ Entity Extraction

Using spaCy with a lightweight EntityRuler to detect academic subjects (e.g., machine learning, calculus, photosynthesis).

✔ Response Engine

Uses intents.json to map each intent to response templates.

✔ REST API

Built with FastAPI to serve predictions for web or mobile clients.

🧱 2. Final Project Directory Structure

chat-bot/
│ main.py
│ rout.py
│ requirements.txt
│ README.md
│
├── data/
│     intents.json
│
├── models/
│     intent_model.pkl
│     vectorizer.pkl
│     spacy_nlp/
│
└── src/
      nlu_trainer.py
      nlu_initializer.py
      nlu_pipeline.py
      services/
          chat_service.py

Each component is modular and replaceable.

🎯 3. Initial Planning & Architecture

The system was originally designed around three NLP components:

1️⃣ Intent Classifier

Detects user intent (greetings, ask_explanation, ask_formula, ask_difference, etc.)

Built using scikit-learn LinearSVC + TF-IDF.

2️⃣ NER Model

Detects subjects/topics in the question

Implemented with spaCy + EntityRuler

Detects entities like:

machine learning

photosynthesis

calculus

3️⃣ Response Engine

Uses templates inside intents.json

Future versions can generate dynamic academic responses.

This design ensures a highly modular system where each component can be improved independently.

📝 4. Dataset Preparation (PHASE 1)

The dataset is stored in:

data/intents.json

Format:

{
  "intents": [
    {
      "intent": "greetings",
      "patterns": ["hi", "hello", ...],
      "responses": ["Hello! How can I help you today?", ...]
    },
    ...
  ]
}

Dataset Highlights:

✔ 27 total intents
✔ ~30 patterns per intent (900+ training samples)
✔ Multiple response templates per intent
✔ All academic categories:
definitions • explanations • differences • examples • formulas • comparison • advantages • disadvantages • etc.

This dataset is the foundation of your NLU engine.

🧠 5. Model Training (PHASE A)

Training is performed using:

python src/nlu_trainer.py

The trainer performs:

Load all training examples

Train TF-IDF vectorizer

Train LinearSVC intent classifier

Create spaCy EntityRuler

Save all models into /models/

Generated models:

models/
│ intent_model.pkl
│ vectorizer.pkl
└── spacy_nlp/

This step must be run before starting the API.

🧩 6. NLU Pipeline (PHASE B)

Two core components are created:

🔹 nlu_initializer.py

Loads:

intent_model.pkl

vectorizer.pkl

spaCy NLP pipeline

Used by ChatService.

🔹 nlu_pipeline.py

Performs:

✔ Intent Classification

intent, confidence = predict_intent(text)

✔ Entity Extraction

entities = extract_entities(text)

✔ Combined NLU Output

{
  "intent": "ask_explanation",
  "confidence": 2.17,
  "entities": [("machine learning", "SUBJECT")]
}

This is the “brain” of the chatbot.

💬 7. Chat Service Logic (PHASE C)

chat_service.py handles full message processing:

✔ Runs NLU
✔ Selects a response from intents.json
✔ Formats final output

Example output:

{
  "sender": "bot",
  "intent": "ask_definition",
  "confidence": 1.85,
  "entities": [],
  "reply": "Here is the definition:"
}

This service keeps all chat-related logic isolated.

🌐 8. API Layer (PHASE D)

rout.py exposes a single POST route:

POST /chat

Supports both:

JSON

form-data

Example request:

{
  "message": "explain gravity",
  "sender": "user"
}

Example response:

{
  "intent": "ask_explanation",
  "confidence": 2.14,
  "entities": [],
  "reply": "Let me explain that:",
  "sender": "bot"
}

main.py starts the API using Uvicorn:

python main.py

🧪 9. Testing & Validation (PHASE E)

Testing is done at multiple levels:

🔹 E1: Test Intent Classification

clf.predict(vec.transform(["give an example of os"]))

Expected → ask_examples

🔹 E2: Test spaCy Entities

nlp("explain machine learning")

Expected → [('machine learning', 'SUBJECT')]

🔹 E3: Test NLUPipeline

nlu.run("difference between virus and bacteria")

🔹 E4: Test ChatService

asyncio.run(chat.process_message("hi"))

🔹 E5: Test API (Postman)

Send POST request to:

http://localhost:8000/chat

Everything should work end-to-end.

📈 10. Future Enhancements (PHASE F-G-H)
🔮 Phase F — Dynamic Academic Answers

Currently, responses are templates.
Upgrade paths:

Add long-form explanations

Generate formulas dynamically

Add algorithm steps

Add code generation

Integrate with Wikipedia APIs

🖥️ Phase G — Frontend UI

You can build a simple web interface using:

HTML + JavaScript

React

Streamlit

Flutter

Frontend interacts with /chat API.

🧠 Phase H — Smarter NER

Enhance NER by adding patterns for:

Topic

Programming languages

Theorems

Formulas

Algorithms

Even migrate to:

Custom spaCy NER

HuggingFace transformer-based NER

🧺 11. Additional Recommendations
✔ Add CORS Middleware

If you plan to connect to frontend.

✔ Add /ping health check

Debug quickly.

✔ Add logging system

Better production monitoring.

✔ Create a fallback knowledge retrieval module

Use Wikipedia or textbooks for deeper academic answers.

🏁 12. Conclusion

You now have:

✔ Fully functional academic chatbot
✔ End-to-end NLU pipeline
✔ Clean FastAPI backend
✔ Accurate intent classification
✔ Basic NER support
✔ Modular, extensible design

This project is structured to scale — you can easily expand intents, add more entities, or swap ML models with minimal changes.