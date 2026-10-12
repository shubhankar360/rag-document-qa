# rag-document-qa

**Answers questions about your documents, cites the passage each answer came
from, and refuses when the documents don't contain one.**

A complete retrieval-augmented generation service — loading, recursive chunking
with overlap, embedding, vector search, a groundedness guard, context assembly,
and citation-carrying answers — behind a CLI, a FastAPI service and a Streamlit
chat UI.

Runs offline on a clean checkout: **no API key, no model download, no vector
database.** Add an Anthropic key and generation upgrades to Claude; the pipeline
around it is unchanged.

```bash
pip install -r requirements.txt
python cli.py sample_docs "how long does a rotated API key stay valid?"
```

```
Rotating a key issues a new secret and keeps the old one valid for twenty four
hours, which lets you deploy the replacement without downtime. [1]

Sources:
  [1] api-guide.md#chunk0 (0.964)
  [2] api-guide.md#chunk1 (0.609)
```

---

## The part most RAG demos skip

A RAG system that always answers is a RAG system that confabulates. Retrieval
returns *something* for every query — ask a support-docs index who won the 1998
World Cup and it will hand the model three passages about billing, and the model
will write a confident paragraph out of them. **The guard has to run before
generation**, and getting it right here took measurement rather than a default.

The obvious approach is a similarity floor: refuse when the top passage scores
below some threshold. Measured on this repo's own sample documents:

| Query | Top cosine |
| --- | --: |
| "how long is raw event data retained?" (relevant) | 0.905 |
| "what is the airspeed velocity of an unladen swallow" (nonsense) | **0.907** |

**The nonsense question scores higher.** Normalising against the corpus doesn't
rescue it either — the top hit's z-margin over the corpus mean spans 1.12–1.78
for on-topic questions and 0.97–1.48 for off-topic ones, which overlap. The
cause is that character-n-gram TF-IDF plus SVD largely encodes generic English
text structure, so every query looks broadly similar to every chunk.

What does separate them is **vocabulary coverage** — asking a lexical question
lexically: do this question's content words appear in the indexed corpus at all?

| | on-topic (8 queries) | off-topic (6 queries) |
| --- | --: | --: |
| Vocabulary coverage | **0.75 – 1.00** | **0.00 – 0.25** |

A threshold at 0.5 sits in a wide empty gap. Matching is prefix-based in both
directions so `refund` matches `refunds` and `retained` matches `retention`
without pulling in a stemmer.

Both query sets are checked in as parametrised tests (`tests/test_grounding.py`),
along with a test that **fails if cosine scores ever become discriminative** —
so if a future embedder change makes the simpler guard viable, the suite says so
instead of leaving the lexical one in place forever.

## What's in the pipeline

```
document → load → chunk (recursive, overlapping, offset-tracked)
                    → embed → vector index
question  → coverage guard → refuse, or
          → retrieve top-k → number the passages → generate → answer + citations
```

**Chunking** splits on the strongest structural boundary that fits — paragraph,
then line, then sentence, then clause, then word — and only hard-cuts when a
single span exceeds the limit with no separator left. Every chunk keeps its
source and character offsets, which is what lets a citation point at an exact
span rather than a whole file. Overlap carries context across boundaries so a
fact that straddles a cut survives intact in at least one chunk.

**Context assembly** numbers each retrieved passage, and those numbers are what
the model is told to cite. The same numbers come back attached to real chunks
with real offsets, so every claim is traceable to text you can go read.

**Generation** is an interface with two implementations. `ClaudeAnswerer` calls
the Anthropic Messages API with a system prompt that restricts the model to the
supplied sources and requires inline citations — and checks `stop_reason` before
reading the response, since a refusal comes back as a successful HTTP 200 with
no text. `ExtractiveAnswerer` is the zero-dependency fallback: it returns the
sentences that best cover the question, with their source markers intact. It is
not a language model and doesn't pretend to be one — it exists so the retrieval
half is verifiable end to end without credentials.

## Two bugs the tests now pin

Both were found by reading actual output rather than by a failing assertion,
which is why each has a regression test named after the behaviour:

- **Sentences were being truncated at line breaks.** Source documents are
  hard-wrapped, so splitting the raw text line by line cut *"...valid for twenty
  four hours"* at `twenty` — the answer was retrieved correctly and then
  destroyed on the way out. Whitespace is now collapsed before sentence
  splitting.
- **Ranking by term density promoted short sentences.** *"Create an API key under
  Settings."* outranked the sentence that actually answered the question,
  because a short sentence sharing one common word scores well on
  hits-per-word. Ranking is now by how much of the *question* a sentence covers,
  with brevity only as a tie-break.

## Install and run

```bash
git clone https://github.com/shubhankar360/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt

python cli.py sample_docs "how long is raw event data retained?"
python cli.py sample_docs               # interactive
pytest -q                               # 65 tests, no network
```

Every extra is optional and each enables exactly one thing:

```bash
pip install anthropic              # generation via Claude
pip install sentence-transformers  # real neural embeddings instead of the LSA default
pip install pypdf                  # PDF ingestion
pip install "fastapi[standard]"    # the HTTP API
pip install streamlit              # the chat UI
```

The pipeline reports which backends it resolved at startup, so it is never
ambiguous whether you are running on Claude or the fallback:

```
indexed 5 chunks from sample_docs (embedder: lsa, answerer: extractive)
```

### HTTP API

```bash
python serve.py --index sample_docs
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how long is raw event data retained?"}'
```

`GET /health` reports chunk count and the resolved backends; `POST /ask` returns
the answer, whether it was grounded, and the citations. `POST /index` is off
unless you start the server with `--index-root <dir>`, and then it only accepts
paths inside that directory (see below for why).

### Deploy to Google Cloud Run

The container indexes `sample_docs` at startup (point `DOCS_PATH` elsewhere for
your own documents) and serves the API on `$PORT`. It runs as a non-root user,
scales to zero when idle, and `max_instances` (2 by default) caps the bill.

```bash
cd deploy/gcp
cp terraform.tfvars.example terraform.tfvars    # set project_id
terraform init && terraform apply               # registry, service, identities, keyless CI auth
terraform output github_variables               # add these four as GitHub repo variables
```

After that, every push to `main` runs the tests, validates the Terraform,
smoke-tests the image, then builds, pushes and deploys it and checks that the
live URL answers a question. GitHub Actions authenticates through Workload
Identity Federation, so there is no service-account key stored anywhere, and
only this repo's `main` branch can deploy. The deployer can push to one image
repository and update one service; it has no project-wide roles.

To use Claude in the deployed service, apply with `enable_claude = true` and add
the key straight to Secret Manager, so it never passes through Terraform state:

```bash
printf %s "$ANTHROPIC_API_KEY" | gcloud secrets versions add anthropic-api-key --data-file=-
```

Two things I found while putting this on the internet:

- **`POST /index` was a file-read hole.** It indexed any path on the server.
  Harmless on localhost; deployed, anyone could index `/etc/passwd`, ask about
  it, and get the contents back as cited excerpts. Indexing over HTTP is now
  opt-in and confined to one directory, with `../`, absolute paths and
  symlinks rejected. CI checks that the published image returns 403 for it.
- **A one-chunk corpus crashed the offline embedder.** The character n-gram
  vectoriser requires each term to appear in at least two chunks, which is
  impossible with one, so a short single-paragraph upload failed. Fixed, with a
  regression test.

Terraform also ignores the service's image after the first apply. CI owns
deploys; without that, the next `terraform apply` would quietly roll the live
service back to the bootstrap image.

### Chat UI

```bash
streamlit run app_streamlit.py
```

Upload documents, ask questions, expand any answer to see the passages it was
built from with their similarity scores.

## Use it as a library

```python
from ragqa import RagPipeline

pipeline = RagPipeline(chunk_chars=900, chunk_overlap=150, top_k=4)
pipeline.index_path("./docs")

answer = pipeline.ask("What is the refund window?")
print(answer.text, answer.grounded)
for citation in answer.citations:
    print(citation.marker, citation.source, citation.score, citation.start, citation.end)
```

Swap any stage — every one is an argument, not a hard-coded import:

```python
from ragqa import ClaudeAnswerer, RagPipeline, VectorStore
from ragqa.embeddings import SentenceTransformerEmbedder

pipeline = RagPipeline(
    store=VectorStore(SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")),
    answerer=ClaudeAnswerer(model="claude-opus-5"),
    min_coverage=0.6,   # stricter refusal
)
```

## Tuning

| Knob | Default | What moves |
| --- | --: | --- |
| `chunk_chars` | 900 | Smaller isolates facts but strips context; larger pads the prompt with irrelevant text you pay for |
| `chunk_overlap` | 150 | Insurance against a fact landing on a chunk boundary |
| `top_k` | 4 | More passages raise recall and prompt cost together |
| `min_coverage` | 0.5 | The refusal threshold — raise it to refuse more readily |
| `min_score` | 0.0 | Similarity floor, off by default (see above); useful with a neural embedder |

## Layout

```
ragqa/
  chunking.py    recursive splitting, overlap, character offsets
  loaders.py     text/markdown built in, PDF when pypdf is installed
  embeddings.py  LSA (offline default) and sentence-transformers backends
  store.py       normalised in-memory index; cosine as one matrix multiply
  grounding.py   the coverage guard, with the measurement that motivated it
  llm.py         Claude and extractive answerers behind one interface
  pipeline.py    index → guard → retrieve → assemble → answer → cite
  api.py         FastAPI adapter (imported lazily)
cli.py           terminal Q&A, one-shot or interactive
serve.py         run the HTTP API
Dockerfile       the image Cloud Run serves
deploy/gcp/      Terraform: Cloud Run, Artifact Registry, Secret Manager, keyless CI auth
app_streamlit.py chat UI
sample_docs/     three documents to try it against
tests/           65 tests — chunking, retrieval, grounding, answering, end to end
```

## Licence

MIT
