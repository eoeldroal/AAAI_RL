# verl documentations

> **Project note.** This directory primarily contains upstream and code-facing
> documentation. StreamWeave paper work starts at
> [`papers_RL/README.md`](papers_RL/README.md). Project-specific `*_RL.md` and
> `DR-*.md` files are reference-only implementation, operation, and experiment
> records; they do not own the paper's current claims or terminology.

## Build the docs

```bash
# If you want to view auto-generated API docstring, please make sure verl is available in python path. For instance, install verl via:
# pip install .. -e[test]

# Install dependencies needed for building docs.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d _build/html/
```
Launch your browser and navigate to http://localhost:8000 to view the documentation. Alternatively you could drag the file `_build/html/index.html` to your local browser and view directly.
