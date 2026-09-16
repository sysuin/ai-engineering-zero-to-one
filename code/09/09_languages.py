# timeout: 900
# Tickets in four languages: what the same meaning costs in tokens, and whether the language of
# the instruction changes the extraction.

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from clarity.config import MODEL_FAST

client = OpenAI()
ENC = tiktoken.get_encoding("o200k_base")

EMAIL = {
    "English": "Good morning. We received our delivery yesterday, but two of the five boxes were damaged and "
               "the cleaning cloths inside were wet. Could you arrange a collection and send replacements before "
               "Friday? Our account number is 40211. Thank you for your help.",
    "French": "Bonjour. Nous avons reçu notre livraison hier, mais deux des cinq cartons étaient endommagés et "
              "les chiffons de nettoyage à l'intérieur étaient mouillés. Pourriez-vous organiser un enlèvement et "
              "nous envoyer des articles de remplacement avant vendredi ? Notre numéro de compte est 40211. "
              "Merci de votre aide.",
    "Spanish": "Buenos días. Recibimos nuestro pedido ayer, pero dos de las cinco cajas estaban dañadas y los "
               "paños de limpieza del interior estaban mojados. ¿Podrían organizar la recogida y enviarnos los "
               "repuestos antes del viernes? Nuestro número de cuenta es 40211. Gracias por su ayuda.",
    "German": "Guten Morgen. Wir haben unsere Lieferung gestern erhalten, aber zwei der fünf Kartons waren "
              "beschädigt und die Reinigungstücher darin waren nass. Könnten Sie eine Abholung veranlassen und "
              "uns bis Freitag Ersatz schicken? Unsere Kundennummer ist 40211. Vielen Dank für Ihre Hilfe.",
}

TICKETS = {  # (category, identifier, text) per language
    "English": [("Delivery", "40752", "The parcel never arrived. Order number 40752."),
                ("Quality", "MRD-SAN-047", "The product MRD-SAN-047 arrived damaged. We need a replacement."),
                ("Billing", "INV-73451", "Invoice INV-73451 was charged twice.")],
    "French": [("Delivery", "40752", "Le colis n'est jamais arrivé. Numéro de commande 40752."),
               ("Quality", "MRD-SAN-047", "Le produit MRD-SAN-047 est arrivé endommagé. Nous avons besoin d'un remplacement."),
               ("Billing", "INV-73451", "La facture INV-73451 a été facturée deux fois.")],
    "Spanish": [("Delivery", "40752", "El paquete nunca llegó. Número de pedido 40752."),
                ("Quality", "MRD-SAN-047", "El producto MRD-SAN-047 llegó dañado. Necesitamos un reemplazo."),
                ("Billing", "INV-73451", "La factura INV-73451 se cobró dos veces.")],
    "German": [("Delivery", "40752", "Das Paket ist nie angekommen. Bestellnummer 40752."),
               ("Quality", "MRD-SAN-047", "Das Produkt MRD-SAN-047 kam beschädigt an. Wir brauchen einen Ersatz."),
               ("Billing", "INV-73451", "Die Rechnung INV-73451 wurde doppelt berechnet.")],
}

INSTRUCTION = {
    "English": "Classify the support ticket as Delivery, Quality or Billing, and copy out the order number, "
               "product code or invoice number it mentions.",
    "French": "Classez le ticket d'assistance dans l'une des catégories Delivery, Quality ou Billing, et recopiez "
              "le numéro de commande, la référence produit ou le numéro de facture qu'il mentionne.",
    "Spanish": "Clasifica el ticket de soporte como Delivery, Quality o Billing, y copia el número de pedido, el "
               "código de producto o el número de factura que menciona.",
    "German": "Ordnen Sie das Support-Ticket einer der Kategorien Delivery, Quality oder Billing zu und schreiben "
              "Sie die darin genannte Bestellnummer, Produktnummer oder Rechnungsnummer ab.",
}


class Ticket(BaseModel):
    category: Literal["Delivery", "Quality", "Billing"]
    identifier: str


english = len(ENC.encode(EMAIL["English"]))
print("the same support email, counted with one tokeniser\n")
print(f"  {'language':10}{'characters':>11}{'tokens':>8}{'vs English':>12}")
for language, text in EMAIL.items():
    n = len(ENC.encode(text))
    print(f"  {language:10}{len(text):>11}{n:>8}{n / english:>11.2f}x")


def extract(language: str, ticket, instruction_language: str) -> bool:
    category, identifier, text = ticket
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=100, response_format=Ticket,
        messages=[{"role": "system", "content": INSTRUCTION[instruction_language]},
                  {"role": "user", "content": text}]).choices[0].message.parsed
    return bool(parsed and parsed.category == category and parsed.identifier == identifier)


RUNS = 3
jobs = [(lang, t, instr) for lang in TICKETS for t in TICKETS[lang]
        for instr in {"English", lang} for _ in range(RUNS)]
with ThreadPoolExecutor(max_workers=12) as pool:
    results = list(pool.map(lambda job: (job, extract(*job)), jobs))

print(f"\nthree tickets per language x {RUNS} runs: category and identifier both right\n")
print(f"  {'ticket in':10}{'English instruction':>21}{'same-language instruction':>27}")
for lang in TICKETS:
    en = [ok for (l, _, i), ok in results if l == lang and i == "English"]
    same = [ok for (l, _, i), ok in results if l == lang and i == lang]
    print(f"  {lang:10}{sum(en):>18}/{len(en)}{(str(sum(same)) + '/' + str(len(same))) if lang != 'English' else '—':>27}")
