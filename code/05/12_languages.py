# The same sentence costs different amounts in different languages, because the tokeniser's
# vocabulary was learned mostly from English.

import tiktoken

from clarity.config import MODEL_FAST

encoder = tiktoken.encoding_for_model(MODEL_FAST)

SENTENCE = {
    "English":  "The supplier raised prices by twelve percent in January.",
    "German":   "Der Lieferant erhöhte die Preise im Januar um zwölf Prozent.",
    "Spanish":  "El proveedor subió los precios un doce por ciento en enero.",
    "Polish":   "Dostawca podniósł ceny o dwanaście procent w styczniu.",
    "Greek":    "Ο προμηθευτής αύξησε τις τιμές κατά δώδεκα τοις εκατό τον Ιανουάριο.",
    "Hindi":    "आपूर्तिकर्ता ने जनवरी में कीमतें बारह प्रतिशत बढ़ा दीं।",
    "Japanese": "サプライヤーは1月に価格を12%値上げした。",
}

base = len(encoder.encode(SENTENCE["English"]))
print(f"  {'language':9} {'chars':>5} {'tokens':>6} {'vs English':>10}")
for language, sentence in SENTENCE.items():
    n = len(encoder.encode(sentence))
    print(f"  {language:9} {len(sentence):>5} {n:>6} {n / base:>9.2f}x")
