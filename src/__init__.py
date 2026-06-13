"""Warden backend package.

Layout note (logged as D48 in docs/decisions.md): the battery-feature
task mandates a `src/` code root. docs/02-architecture.md sketched the
same modules under `app/`; the mapping is 1:1 (app/engine -> src/engine,
app/main.py -> src/api/main.py). `src/` is the layout of record.
"""
