# timeout: 600
# What LCEL is nice at, and what it costs you the first time something breaks.

import json
import sys
import traceback
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402

DOC = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md").read_text()[:1200]

# ------------------------------------------------------------------ the nice part
prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarise business documents in one sentence."),
    ("user", "{document}")])
chain = prompt | ChatOpenAI(model=MODEL_FAST, temperature=0) | StrOutputParser()

print("--- LCEL ---")
print(chain.invoke({"document": DOC}).strip()[:160])
print()
print("Three objects and two pipes. That is genuinely less typing than the SDK, and the")
print("pieces are swappable: change the model line and nothing else moves.")


# ------------------------------------------------------------------ the other part
def frames(error: BaseException) -> tuple[int, int]:
    """How much of this traceback is code you could edit?"""
    stack = traceback.extract_tb(error.__traceback__)
    theirs = sum(1 for f in stack if "site-packages" in f.filename)
    return len(stack) - theirs, theirs


print("\n--- the same mistake, twice ---")
print("A template variable is misspelled: {docment} instead of {document}.\n")

broken = (ChatPromptTemplate.from_messages([("user", "{docment}")])
          | ChatOpenAI(model=MODEL_FAST) | StrOutputParser())
try:
    broken.invoke({"document": DOC})
except Exception as error:                                   # noqa: BLE001
    mine, theirs = frames(error)
    print(f"LCEL       : {type(error).__name__}")
    print(f"             {str(error).splitlines()[0][:90]}")
    plural = "frame" if mine + theirs == 1 else "frames"
    print(f"             {mine + theirs} {plural}, {theirs} of them inside "
          f"site-packages")
    lcel = (mine, theirs)

client = OpenAI()
try:
    payload = {"document": DOC}
    client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": payload["docment"]}])
except Exception as error:                                   # noqa: BLE001
    mine, theirs = frames(error)
    print(f"plain SDK  : {type(error).__name__}")
    print(f"             {str(error).splitlines()[0][:90]}")
    plural = "frame" if mine + theirs == 1 else "frames"
    print(f"             {mine + theirs} {plural}, {theirs} of them inside "
          f"site-packages")
    sdk = (mine, theirs)

print()
print("Be fair about which message is better: LangChain's names the variable it wanted")
print("and the one it got, and the SDK's is the bare word 'docment'. On wording, the")
print("framework wins.")
print()
print(f"The cost is the shape of the traceback. The SDK raises {sdk[0]} frame deep, in")
print(f"the line you typed. LCEL raises {lcel[0] + lcel[1]} frames deep, {lcel[1]} of "
      f"them in code you did not")
print("write, and the failure is reported in the framework's vocabulary - a prompt")
print("variable - rather than in yours.")
print()
print("That is the trade, and it is not an argument against frameworks. It is the")
print("argument for hand-building first: you can read a stack trace through someone")
print("else's abstraction only once you know what it is doing on your behalf.")
