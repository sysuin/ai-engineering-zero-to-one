# Prompt templates: three ways to fill in the blanks, and the ways each one breaks.

import string

document = 'The Supplier shall give notice. Example payload: {"days": 30}'
TEMPLATE_BRACES = "Extract the notice period as JSON like {\"days\": 0}.\n\n{document}"

# 1. str.format treats every brace as a placeholder — including the JSON in the prompt.
try:
    print(TEMPLATE_BRACES.format(document=document))
except (KeyError, ValueError, IndexError) as error:
    print(f"str.format failed: {type(error).__name__}: {error}")

# The fix is to double literal braces in the template: {{ and }}.
print(TEMPLATE_BRACES.replace('{"', '{{"').replace("0}", "0}}").format(document=document)[:60],
      "...")

# 2. string.Template uses $names, which almost never appear in prose or JSON — but a
#    missing value is silently left in the prompt by safe_substitute.
template = string.Template("Summarise $document for $audience.")
print("\nsafe_substitute with a value missing:")
print("  ", template.safe_substitute(document="the Q3 review"))


# 3. A render function that refuses to send a prompt with a hole in it.
def render(text: str, **values: str) -> str:
    """Fill ${name} placeholders; fail on any missing or unused value."""
    t = string.Template(text)
    needed = {m.group("named") or m.group("braced") for m in t.pattern.finditer(text)
              if m.group("named") or m.group("braced")}
    missing, unused = needed - values.keys(), values.keys() - needed
    if missing or unused:
        raise ValueError(f"missing {sorted(missing)}; unused {sorted(unused)}")
    return t.substitute(values)


print("\nrender() with everything supplied:")
print("  ", render("Summarise $document for $audience.", document="the Q3 review",
                   audience="the board"))
try:
    render("Summarise $document for $audience.", document="the Q3 review", audince="board")
except ValueError as error:
    print("render() with a typo in a name:", error)
