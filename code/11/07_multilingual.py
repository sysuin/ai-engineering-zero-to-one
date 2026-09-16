# The same ten questions asked in three other languages, against the English contracts.

import numpy as np
from openai import OpenAI

from _corpus import QUERIES, load
from clarity.config import MODEL_EMBED

client = OpenAI()
corpus = load()

TRANSLATIONS = {
    "Spanish": ["¿Cuánto tiempo tenemos para pagar una factura?",
                "¿Cuál es el límite de una subida de precios?",
                "¿Podemos rescindir este contrato antes de tiempo?",
                "¿Qué pasa si la mercancía llega rota?",
                "¿Cuánto podemos reclamar si nos causan una pérdida?",
                "¿Tienen que cumplir un objetivo de entregas?",
                "¿Los tribunales de qué país resuelven una disputa?",
                "¿Qué nos venden exactamente?",
                "¿Con cuánta antelación deben avisar antes de subir precios?",
                "¿Hay alguna penalización por retrasos repetidos en los envíos?"],
    "German":  ["Wie lange haben wir Zeit, eine Rechnung zu bezahlen?",
                "Wie hoch darf eine Preiserhöhung höchstens sein?",
                "Können wir diesen Vertrag vorzeitig beenden?",
                "Was passiert, wenn die Ware beschädigt ankommt?",
                "Wie viel können wir fordern, wenn sie uns einen Schaden verursachen?",
                "Müssen sie ein Lieferziel erreichen?",
                "Die Gerichte welchen Landes entscheiden einen Streit?",
                "Was verkaufen sie uns eigentlich?",
                "Wie lange im Voraus müssen sie eine Preiserhöhung ankündigen?",
                "Gibt es eine Strafe für wiederholt verspätete Lieferungen?"],
    "French":  ["Combien de temps avons-nous pour régler une facture ?",
                "Quel est le plafond d'une hausse de prix ?",
                "Pouvons-nous résilier ce contrat de manière anticipée ?",
                "Que se passe-t-il si la marchandise arrive cassée ?",
                "Combien pouvons-nous réclamer s'ils nous causent une perte ?",
                "Doivent-ils atteindre un objectif de livraison ?",
                "Les tribunaux de quel pays tranchent un litige ?",
                "Que nous vendent-ils exactement ?",
                "Avec quel préavis doivent-ils annoncer une hausse de prix ?",
                "Y a-t-il une pénalité pour des retards de livraison répétés ?"],
}


def embed(texts):
    data = client.embeddings.create(model=MODEL_EMBED, input=texts).data
    v = np.array([d.embedding for d in data])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


V = embed([c["text"] for c in corpus])
want = [w for _, w in QUERIES]
english = embed([q for q, _ in QUERIES])


def recall_at_1(Q):
    return sum(corpus[int(np.argmax(V @ q))]["clause"] == w for q, w in zip(Q, want)) / len(want)


print(f"  {'language':9} {'recall@1':>9} {'similarity to the English question':>36}")
print(f"  {'English':9} {recall_at_1(english):>9.0%}")
for language, questions in TRANSLATIONS.items():
    Q = embed(questions)
    agreement = float(np.mean(np.sum(Q * english, axis=1)))
    print(f"  {language:9} {recall_at_1(Q):>9.0%} {agreement:>36.3f}")
