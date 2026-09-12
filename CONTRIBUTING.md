# Clarity — notes for whoever works on this next

This file is for anyone about to change this repository. It says the things that are
true about this repository but not visible in any one file.

## What this is

Clarity answers questions about Meridian — a fictional mid-size company — using its
documents and its data warehouse. It is the worked project of *AI Engineering from Zero
to One*, built across 22 milestones, and the version you should read is
`code/clarity/v1_0/`. The earlier `v0_*` directories are the book's history: each one is
the system as it stood at the end of a chapter, kept so a reader can see what changed.

**Do not refactor the `v0_*` directories.** They are not duplication to be cleaned up;
they are the illustrations. If a bug exists in `v0_6`, it exists in Chapter 14 too, and
fixing one without the other breaks the book.

## Where things are

```
code/clarity/v1_0/      the system as it stands: clarity.py, service.py, cli.py, demo.py
code/clarity/platform/  the cross-cutting pieces: tracing, replay, resilience, gateway,
                        cache, guard — none of which know about each other
code/clarity/evals/     the golden set, the judge, the six-layer suite, the red team
code/clarity/config.py  the ONLY file that names a model. Change it, not the chapters.
code/meridian/          the dataset generator
code/NN/                the executed listings for chapter NN, with their captured .out
```

## The rules this repository is built on

**No output in the book was typed by a human.** `code/_runner.py` executes every listing
and captures its real stdout. If you change a listing, re-run it. If you change a module
a listing imports, *touch the listing* — the cache keys on the listing file's hash, not
its imports, so a shared-module change will not re-run its dependents on its own.

**No model name or price appears in a chapter.** They live in `clarity/config.py` and
Appendix C. Chapters say `MODEL_FAST`.

**No secret appears in any file git can see.** Keys come from `.env`, which is ignored.
Never print a key; print a fingerprint if you must prove one is loaded.

**Prose that states a direction or a magnitude must compute it.** A sentence saying
"the agent was slower" is a bug waiting for the next run to flip. Derive it in the
listing and print the sentence, or write the prose so it survives either outcome. This
is the single most common way a chapter goes wrong.

**One eval scorer, one refusal list.** `clarity/evals/runner.py` owns both. A second copy
is how a system comes to refuse in production and score zero in the suite.

## Running it

`make help` lists everything. The four you will use:

```
make index               build the corpus and its vector index (this one costs money)
make ask Q="..."         one question, with sources and cost
make eval                the six-layer suite against the 120-case golden set
```

## Known weaknesses, deliberately

Chapter 31 reviews this system and finds three. They are real and they are not on a
backlog — they are the price of a system small enough to read:

1. **The index is rebuilt, never updated.** No incremental path exists. `documents.sha256`
   is there so one can be added.
2. **One tenant, structurally.** Two files declare a tenant column; none enforce it. The
   test in §26.2 catches an instance; row-level security would remove the class.
3. **The eval set is 120 cases from one corpus**, 62% of one kind. At that size a 95%
   interval is thirteen points wide, which cannot see a three-point improvement.

If you are about to add a feature, check first whether you are about to build on top of
one of these.

## What to be careful about

- **The golden set is verified, not generated.** Adding a case means checking the answer
  against the source, not asking a model for one.
