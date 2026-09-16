# Dates and times: the bug every analyst meets eventually, usually at a quarter boundary.

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CHICAGO = ZoneInfo("America/Chicago")    # Meridian's Midwest office

# A date has no time and no timezone. Most of Meridian's warehouse is dates.
opened = date.fromisoformat("2024-08-14")
closed = date.fromisoformat("2024-09-02")
print("opened:", opened, " quarter:", (opened.month - 1) // 3 + 1)
print("days open:", (closed - opened).days)
print("30 days later:", opened + timedelta(days=30))

# A timestamp from an API or a log is almost always UTC. It says an instant, not a
# wall-clock time anywhere in particular.
stamp = datetime.fromisoformat("2024-10-01T03:30:00+00:00")
local = stamp.astimezone(CHICAGO)
print("\nUTC:    ", stamp, " quarter", (stamp.month - 1) // 3 + 1)
print("Chicago:", local, " quarter", (local.month - 1) // 3 + 1)
print("-> the same order is Q4 in UTC and Q3 in the office that took it")

# A 'naive' datetime has no timezone at all. Python refuses to compare it with an aware
# one, which is the right thing to do and still surprises everyone once.
naive = datetime(2024, 10, 1, 3, 30)
try:
    print(naive < stamp)
except TypeError as error:
    print("\nTypeError:", error)

# Daylight saving. Clocks in Chicago went forward at 02:00 on 10 March 2024.
before = datetime(2024, 3, 9, 9, 0, tzinfo=CHICAGO)
one_day = before + timedelta(days=1)                     # same wall-clock time next day
elapsed = one_day.astimezone(timezone.utc) - before.astimezone(timezone.utc)
print("\nbefore:           ", before)
print("+ timedelta(days=1):", one_day)
print("hours that actually passed:", elapsed.total_seconds() / 3600)

# The rule that avoids nearly all of this: store and compute in UTC, convert to a local
# time only to show a person or to decide a business boundary, and say which you did.
now_utc = datetime.now(timezone.utc)
print("\naware 'now' has tzinfo:", now_utc.tzinfo is not None)
