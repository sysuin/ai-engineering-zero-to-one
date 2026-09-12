"""
Clarity v0.21 — the defences that do not depend on the model noticing.

§29.3 measures what a well-worded instruction is worth against a well-written
injection: something, and not enough. Everything here works whether or not the model
was fooled, which is the property that makes it a control rather than a hope.

    strip_active     remove the constructs an injection needs to act: image URLs,
                     links to hosts you do not know, HTML comments
    fence            mark data as data structurally, so "ignore the above" has an
                     above that is not your instructions
    allowlist        an action the model may request is one you have written down
    egress           what may leave: no unknown host, ever, in any rendered output
"""
from __future__ import annotations

import re

IMAGE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
COMMENT = re.compile(r"<!--.*?-->", re.S)
URL = re.compile(r"https?://([^/\s)\"']+)", re.I)


def strip_active(text: str) -> str:
    """
    Remove what an injected instruction needs in order to do anything.

    An image is the dangerous one and the least obvious: a rendered markdown image
    is an HTTP GET to a host of the author's choosing, with whatever the model wrote
    in the query string. Nobody clicks anything. §29.3.
    """
    text = COMMENT.sub("", text)
    text = IMAGE.sub("[image removed]", text)
    return text


def egress_violations(text: str, allowed: set[str]) -> list[str]:
    """Every host the answer would contact, minus the ones you have approved."""
    return sorted({host.lower() for host in URL.findall(text)
                   if host.lower() not in allowed})


def fence(passages: list[str]) -> str:
    """
    Separate instructions from data structurally rather than by asking nicely.

    The delimiter is not magic — a model can still be persuaded — but it gives the
    prompt a true statement to make: everything between these markers is a quotation
    from a document, and documents do not issue instructions. Combined with stripping,
    it is the difference between "please ignore" and "there is nothing to obey".
    """
    parts = []
    for n, passage in enumerate(passages, start=1):
        clean = strip_active(passage).replace("<<<", "").replace(">>>", "")
        parts.append(f"<<<DOCUMENT {n}>>>\n{clean}\n<<<END DOCUMENT {n}>>>")
    return "\n\n".join(parts)


class Allowlist:
    """
    An action the model may request is one somebody wrote down.

    The alternative — free-form actions, a shell tool, arbitrary SQL — is a system
    whose blast radius is the union of everything its credentials can do, decided at
    runtime by text from a document.
    """

    def __init__(self, actions: dict[str, set[str]]) -> None:
        self.actions = actions
        self.blocked: list[tuple[str, str]] = []

    def permits(self, role: str, action: str) -> bool:
        allowed = action in self.actions.get(role, set())
        if not allowed:
            self.blocked.append((role, action))
        return allowed
