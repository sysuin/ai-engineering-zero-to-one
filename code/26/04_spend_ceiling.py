# A daily spend ceiling for a tenant, under concurrent requests. Checking the total after
# each call lets a burst through; reserving the worst case before the call does not.

import random
import threading
import time

CEILING = 0.50                      # dollars a day for this tenant (illustrative)
WORST_CASE = 0.06                   # the most one request can cost: its token budget, priced
REQUESTS = 30


class Ledger:
    def __init__(self):
        self.lock, self.spent, self.reserved = threading.Lock(), 0.0, 0.0


def run_model(rng: random.Random) -> float:
    time.sleep(0.05)                                 # the call takes a while
    return round(rng.uniform(0.02, WORST_CASE), 4)   # what it actually cost


def check_after(ledger: Ledger, rng: random.Random, served: list) -> None:
    with ledger.lock:
        if ledger.spent >= CEILING:
            return                                   # refused
    cost = run_model(rng)                            # everyone who passed is now spending
    with ledger.lock:
        ledger.spent += cost
    served.append(cost)


def reserve_before(ledger: Ledger, rng: random.Random, served: list) -> None:
    with ledger.lock:
        if ledger.spent + ledger.reserved + WORST_CASE > CEILING:
            return                                   # refused before any money moves
        ledger.reserved += WORST_CASE
    cost = run_model(rng)
    with ledger.lock:
        ledger.reserved -= WORST_CASE                # release the reservation...
        ledger.spent += cost                         # ...and record what it really cost
    served.append(cost)


print(f"ceiling ${CEILING:.2f}; {REQUESTS} requests arrive together; "
      f"each costs up to ${WORST_CASE:.2f}\n")
for name, handler in (("check the total after the call", check_after),
                      ("reserve the worst case before", reserve_before)):
    ledger, served = Ledger(), []
    threads = [threading.Thread(target=handler, args=(ledger, random.Random(i), served))
               for i in range(REQUESTS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    over = ledger.spent - CEILING
    print(f"  {name:32} served {len(served):>2}   spent ${ledger.spent:.2f}   "
          f"{'over the ceiling by $' + format(over, '.2f') if over > 0 else 'within the ceiling'}")
