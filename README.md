# CPS Property Evidence Lab

A clean Streamlit research prototype for the exact CPS formal-property analysis discussed in the project meeting.

## Fixed research goal

1. Curate the consolidated CPS incident dataset.
2. Fetch evidence from URL1–URL4.
3. Classify each incident against the exact 18-property taxonomy.
4. Keep all CONFIRMED labels.
5. Study prevalence within each taxonomy box and correlations within/across boxes, including the complete 18×18 matrix.

## No paid API and no TF-IDF

This version uses only local/public open-source models downloaded from Hugging Face at runtime:

- Semantic evidence retrieval: `sentence-transformers/all-MiniLM-L6-v2`
- Default NLI classifier: `cross-encoder/nli-MiniLM2-L6-H768`
- Optional stronger NLI classifier: `cross-encoder/nli-deberta-v3-base`

There is no OpenAI key, no paid inference API, and no TF-IDF classifier.

The method is deliberately an **evidence-aware local Transformer/NLI pipeline**, rather than a generative chatbot. For each formal property it retrieves the most semantically relevant source passages and asks a Natural Language Inference model whether the passage entails one of four consequence statements: CONFIRMED, POTENTIAL, UNAFFECTED, or CLAIMED. Ambiguous cases become UNKNOWN.

## Pages

### 1. Incident Database
Browse/filter 1,207 incidents, add/edit/delete records in the current Streamlit session, and export the edited CSV.

### 2. Evidence Fetcher
For URL1–URL4:

- direct public HTTP/HTTPS fetch
- HTML extraction with Trafilatura, BeautifulSoup fallback
- PDF extraction with PyMuPDF
- optional Internet Archive Wayback fallback if direct retrieval fails
- explicit FAILED/BLOCKED audit state
- manual pasted evidence fallback
- evidence CSV import/export

### 3. Local Transformer Classifier

- local semantic retrieval (MiniLM embeddings)
- local NLI cross-encoder
- exact 18 properties
- multi-label output
- evidence passage, source and URL retained
- statuses: CONFIRMED / POTENTIAL / UNAFFECTED / CLAIMED / UNKNOWN
- only CONFIRMED labels are used for the research statistics
- classification CSV import/export

### 4. Correlation Explorer
Implements the analysis requested in the discussion:

- 3×3 high-level dimension Phi correlation and co-occurrence
- Functional Correctness: prevalence + 5×5 matrices
- Information Protection: prevalence + 7×7 matrices
- Operational Assurance: prevalence + 6×6 matrices
- cross-box 5×7, 5×6, and 7×6 matrices
- full 18×18 Phi matrix
- full 18×18 co-occurrence matrix
- strongest pairwise associations with Phi, odds ratio, Fisher p-value and Benjamini–Hochberg FDR q-value
- sector and year filters

## Scientific caution

The local NLI models are general-language models, not trained specifically on this CPS taxonomy. The thresholds in `model_engine.py` are initial operating thresholds, not claimed optimal values. Before publication, create a manually labelled validation subset and tune/validate thresholds against it. This is essential if the classifier is used to support empirical claims.

## GitHub / Streamlit

Upload the contents of this folder to the repository root. The Streamlit main file remains:

```text
app.py
```

No secrets are required.

The first classification run downloads the selected Hugging Face models, so it will be slower than later runs. Start with 1–5 incidents and the smaller NLI model.

## Persistence

Streamlit Community Cloud is not a permanent database. Evidence/classification work is held in the current session. Use the page download buttons to save `evidence_state.csv`, `classification_results.csv`, and the edited `incidents.csv`, then import them in a later session or commit the canonical CSV to GitHub.


## 2026-08-26 FIX: all results were UNKNOWN

The earlier decision wrapper was too conservative. It used semantic similarity as a hard gate,
high status thresholds, and a required margin between competing status hypotheses. A property could
therefore have a high NLI score and still be forced to `UNKNOWN`.

This version changes the design:

1. sentence embeddings are **retrieval only**;
2. short direct factual NLI hypotheses are used for the 18 properties;
3. the NLI probabilities (entailment / neutral / contradiction) are retained for diagnosis;
4. negation, potential/capability language, and unverified-claim language are status safeguards;
5. automatic URL fetching is enabled directly on the classifier page.

## Free URL cascade

The fetch cascade is now:

```text
Direct HTML/PDF
   ↓ failure
Jina Reader basic/free (no key)
   ↓ failure
Wayback snapshot
   ↓ failure
FAILED → manual evidence
```

The project does not require a paid API key.


## v3 current-run isolation and incident relevance

This version fixes two issues found during live testing:

- The classifier page previously displayed the entire session evidence table, so old URL3/URL4 and other incidents appeared even when a run selected only three incidents and two URLs.
- Previously saved evidence was also automatically mixed into classification.

v3 now uses exactly URL1..URLN for the selected incidents in the current run. Matching successful evidence can be reused, but only for those allowed URL keys.

A second fix adds **incident relevance** to fetched evidence. Fetched pages often contain background paragraphs about other cyber incidents. Such background text must not become evidence for the selected incident. Transformer embeddings therefore score each fetched chunk against the incident name, attack details, and verified impact summary before NLI classification.
