# Reading a file, and what a path actually is.

from pathlib import Path

review = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md")

print("Does it exist?", review.exists())
print("Just the file name:", review.name)
print("The folder it is in:", review.parent)
print("Size in bytes:", review.stat().st_size)

text = review.read_text()
lines = text.split("\n")

print("\nLine count:", len(lines))
print("First three lines:")
for line in lines[:3]:
    print("   ", line)

# Find every line that mentions Halloway — the account that did not renew.
print("\nLines mentioning Halloway:")
for line in lines:
    if "Halloway" in line:
        print("   ", line.strip()[:78], "...")
