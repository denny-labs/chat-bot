# chat-bot
building a basic chat bot using RASA.


Phase A-
Impact files: nlu_trainer.py
Tain the model and store it in the models.

Phase B-
Impact files: nlu_initializer.py, nlu_pipeline.py
Helps initialize and train the bot.

Phase C-
Impact files: chat_service.py
To enable the proper chatting facility to identify the intent and entities.

Phase E-
Impact files: models folder.
Train the model and store them in seperate files like intent_model.pkl, vectorizer.pkl, spacy_nlp and so on.

code used:

E1 — Test the Intent Classifier Alone

from joblib import load

vec = load("models/vectorizer.pkl")
clf = load("models/intent_model.pkl")

text = "explain machine learning"
X = vec.transform([text])
print(clf.predict(X))

##########################################################################################

E2 — Test spaCy Entity Extraction Alone

import spacy
nlp = spacy.load("models/spacy_nlp")

doc = nlp("explain machine learning in simple terms")
print([(ent.text, ent.label_) for ent in doc.ents])

##########################################################################################

E3 — Test the NLUPipeline Directly

from src.nlu_initializer import NLUInitializer
from src.nlu_pipeline import NLUPipeline

init = NLUInitializer()
models = init.load_all()
nlu = NLUPipeline(models["intent_model"], models["vectorizer"], models["spacy_nlp"])

print(nlu.run("give an example of algorithm"))

##########################################################################################

E4 — Full ChatService Pipeline Test

import asyncio
from src.services.chat_service import ChatService

chat = ChatService()

asyncio.run(chat.process_message("explain gravity"))

##########################################################################################

