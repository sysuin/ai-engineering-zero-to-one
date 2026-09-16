# print() is for the person running a script. logging is for the person reading what
# happened afterwards — often you, at a worse time.

import csv
import logging
import sys

# One line of setup, once, at the top of the program. Real services add %(asctime)s;
# it is left out here so the book's captured output does not change on every run.
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(levelname)-8s %(name)s: %(message)s")
log = logging.getLogger("meridian.load")

log.debug("opening file")                    # below INFO: not shown
with open("data/meridian/warehouse/transactions.csv", newline="") as f:
    rows = list(csv.DictReader(f))
log.info("loaded %d rows from %s", len(rows), "transactions.csv")
large = sum(1 for r in rows if int(r["qty"]) > 100)
log.warning("%d lines have a quantity over 100; worth checking for unit errors", large)

try:
    ratio = 1 / 0
except ZeroDivisionError:
    log.exception("margin for an empty group")   # the message AND the traceback

# Change the level and the same code tells you more, without editing a single call.
logging.getLogger().setLevel(logging.DEBUG)
log.debug("now DEBUG messages appear too")

# Each module gets its own named logger, so you can turn one part up and leave the rest.
logging.getLogger("meridian.api").setLevel(logging.ERROR)
logging.getLogger("meridian.api").info("this is suppressed")
logging.getLogger("meridian.api").error("this is not")
