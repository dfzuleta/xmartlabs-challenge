# DESIGN_AND_OPS.md

## Overview

This document covers the evaluation framework implemented, system improvements made, findings from experimentation, known limitations, and the production roadmap for deploying this aviation RAG system to a university with thousands of student pilots.

---

## Task 1 — Evaluation Framework

### Approach

Three metrics were implemented in `src/evaluation.py`, each measuring a different aspect of the RAG pipeline:

| Metric | Technique | What it measures |
|---|---|---|
| Retrieval Accuracy | Avg cosine similarity (FAISS embeddings) | Are the retrieved chunks relevant to the question? |
| Answer Relevance | NLI CrossEncoder (DeBERTa-v3-small) | Does the answer address what was asked? |
| Faithfulness | LLM-as-judge (Falcon YES/NO via chat template) | Does the answer stay within what the documents say? |

### Evaluation Dataset

11 questions were curated:
- **9 aviation questions** covering maneuvers explicitly described in the FAA Airplane Flying Handbook (lazy 8, steep turn, steep spiral, attitude flying, coordination)
- **2 out-of-domain questions** (George Washington's age, water composition) to test the system's ability to reject irrelevant queries

### Key Design Decisions

**Retrieval uses AVG, not MAX:** Averaging similarity across all 5 chunks penalizes noisy retrievals. A single good chunk surrounded by irrelevant ones should not pass — the system must retrieve consistently relevant content.

**Short-circuit when retrieval fails:** If the max chunk score is below 0.4 (the pipeline threshold), relevance and faithfulness are automatically set to 0. There is no point evaluating answer quality when the system had no relevant information to begin with.

**NLI for relevance, not semantic similarity:** Semantic similarity would score high if the answer uses similar words to the question, even if it doesn't actually answer it. NLI detects whether the answer entails a response to the question — catching contradictions and deflections that similarity would miss.

**LLM-as-judge using Falcon itself:** Faithfulness is evaluated by asking Falcon (the same model used for generation) to judge YES/NO whether the answer is supported by the chunks. Using the chat template format (system + user roles) was critical — raw prompts without the template caused Falcon to ignore the YES/NO instruction.

**Evaluator uses RAGAgent, not a separate pipeline:** `_get_answer()` calls `RAGAgent.act()` directly, evaluating the exact same path the real user sees. An earlier version called `rag_pipeline._gen` directly with a custom prompt — that was measuring a different system than what users experienced.

### Results

**Aviation questions (9):**

| Metric | Score | Result |
|---|---|---|
| Retrieval | 0.57 – 0.70 avg | 9/9 OK |
| Relevance | 0.67 – 0.99 avg | 9/9 OK |
| Faithfulness | 1.000 avg | 9/9 OK |

**Out-of-domain questions (2):**

| Metric | Score | Result |
|---|---|---|
| Retrieval | 0.16 – 0.28 | 2/2 FAIL (correct) |
| Relevance | 0.000 | 2/2 FAIL (correct) |
| Faithfulness | 0.000 | 2/2 FAIL (correct) |

### Known Limitations

**NLI and long answers:** The CrossEncoder model (`cross-encoder/nli-deberta-v3-small`) has a ~512 token input limit. When Falcon generates long procedural answers (step-by-step maneuvers), the NLI score may drop paradoxically — not because the answer is wrong, but because the model struggles with long hypotheses. Larger NLI models exist but add memory and latency costs.

**Relevance does not detect hallucination:** NLI scores high when the answer *addresses* the question, even if the content is incorrect. A hallucinated answer that sounds on-topic will pass Relevance. Faithfulness is the metric that catches this.

**LLM-as-judge is non-deterministic:** Falcon's YES/NO judgment varies slightly across runs. A more reliable alternative would be NLI applied as `(chunks → answer)` entailment, which would make Faithfulness fully deterministic. This is the recommended next step.

### Future Evaluation Work

- **Ground truth dataset:** Create a gold-standard set of (question, expected answer) pairs reviewed by an aviation expert. Enables ROUGE/BERTScore comparison.
- **A/B testing framework:** Log retrieval scores, chunk IDs, and answer quality per session. Compare configurations (chunking strategy, embedding model, prompt version) statistically.
- **Human evaluation loop:** For production, have flight instructors rate a random sample of answers weekly. Automated metrics catch regressions, human evaluation catches nuanced errors.
- **Deterministic faithfulness:** Replace the LLM judge with NLI `(context, answer)` entailment for consistent, reproducible scores.

---

## Task 2 — System Improvements

### Bugs Fixed

**Bug 1 — generate_answer used only 1 chunk (`src/rag.py`)**
The `generate_answer()` method returned `snippets[0]` — only the top-ranked chunk. Falcon received 1/5th of the available context. Fixed to join all 5 chunks. Impact: measurably richer answers on procedural questions.

**Bug 2 — No relevance threshold (`src/rag.py`)**
`retrieve()` always returned 5 chunks regardless of score. Out-of-domain questions (George Washington, water composition) received aviation chunks with scores of 0.15–0.28. Falcon then hallucinated answers. Fixed by adding a score threshold of 0.4 — below it, `NO_INFO` is returned and Falcon says "I don't have enough information."

**Bug 3 — Dead tool in MCP server (`src/mcp_server.py`)**
A `generate_answer` tool handler existed in `_handle_call_tool` but was never declared in `_handle_list_tools`. It was unreachable by any client. Additionally, it called `rag_pipeline.generate_answer(question)` with the wrong signature. Removed.

**Bug 4 — Stale string comparison (`src/mcp_server.py`)**
`search_aviation` compared against the hardcoded string `"I do not know"`. After the pipeline was updated to use the `NO_INFO` constant, this check never fired. Fixed to import and use `NO_INFO`.

**Bug 5 — Character-based chunking destroyed paragraph structure (`src/document_processor.py`)**
`_clean_text()` collapsed all whitespace (including paragraph breaks) to single spaces before chunking. Then `chunk_text()` cut every 500 characters regardless of sentence or paragraph boundaries — frequently mid-sentence. Fixed by preserving paragraph breaks and splitting semantically: by paragraph first, by sentence if a paragraph exceeds the size limit.

**Bug 6 — Follow-up questions failed retrieval (`src/agents.py`)**
When a user asked "And how do I perform it?" after "What is a steep turn?", FAISS searched only the raw follow-up question. The pronoun "it" had no referent — retrieval returned irrelevant chunks and Falcon said it had no information. Fixed by enriching the retrieval query with the previous user turn before sending to FAISS.

### Improvements Made

**Falcon removed from `VectorStoreRAGPipeline`**
The pipeline loaded Falcon (2GB) but only called it in a rare fallback — when FAISS returned no results. In practice this never triggered because FAISS always returns 5 chunks. The fallback path fed "I do not know" to Falcon, which then hallucinated. The pipeline now handles the no-results case with a direct message, and generation responsibility belongs entirely to `RAGAgent`.

**Prompt engineering (`src/agents.py`)**
Three changes to the system prompt:
1. Explicit instruction to answer ONLY from the provided context — reduces hallucination
2. Few-shot examples showing correct aviation response and out-of-domain rejection — guides Falcon's format and behavior
3. Format instruction (2-3 paragraphs) — produces more consistent, readable responses

Before these changes, Relevance averaged 0.689. After: 0.854. Faithfulness went from 2/9 to 9/9.

**Semantic chunking (`src/document_processor.py`)**
Paragraph-based chunking produces more coherent chunks — complete ideas instead of sentence fragments. This gives Falcon richer, more interpretable context per chunk.

### Model Limitations Observed

**Falcon3-1B struggles with selective extraction:** When the retrieved chunks contain mixed information (e.g., advantages and disadvantages in the same passage), Falcon tends to repeat the most frequent content rather than filtering to what was specifically asked. In testing, asking "What are the disadvantages of turboprop engines?" after "What are the advantages?" returned largely the same content — the model could not isolate the disadvantages from a mixed paragraph.

This is a size constraint. A 7B+ parameter model (Falcon 7B, Llama 3 8B, Mistral 7B) would handle this distinction significantly better. The trade-off is memory (7B models require ~16GB GPU RAM vs ~2GB for 1B) and latency.

**Mitigation without changing the model:** Better chunking that separates advantages/disadvantages into distinct chunks would partially address this. Section-aware chunking (detecting headers like "Advantages" and "Disadvantages" in the PDF) would allow FAISS to retrieve more targeted content.

### What Was Not Implemented (Future Work)

**Semantic query rewriting:** Instead of appending the previous question to the retrieval query (current approach), use Falcon to rewrite the follow-up question into a standalone query before searching. More accurate for multi-turn conversations.

**Embedding model upgrade:** `all-MiniLM-L6-v2` is fast but limited. `all-mpnet-base-v2` or domain-specific aviation embeddings would improve retrieval quality, especially for technical terminology.

**Re-ranking:** After FAISS retrieval, apply a CrossEncoder re-ranker to reorder the top-K chunks by relevance to the question. Improves precision of the context Falcon receives.

**MCP chat on Windows:** The `chat` command (MCP-based) gets stuck on Windows due to blocking Falcon generation inside an asyncio event loop combined with stdio subprocess communication. Workaround implemented as `direct-chat` command. Root fix requires running `agent.act()` in a thread executor and extending the MCP client's connection wait time to account for Falcon loading.

---

## Task 3 — Production Roadmap

The system currently runs as a local Python process serving one user at a time. Productionalizing it for thousands of concurrent student pilots requires changes across every layer.

### Scalability

**Current state:** Single Python process, Falcon on CPU, one request at a time.

**Required changes:**
- Move Falcon generation to GPU instances (A10G or T4). On CPU, generation takes 1–5 minutes per response. On GPU, under 5 seconds.
- Decouple retrieval from generation. FAISS search is fast (~50ms); Falcon generation is slow. Use a request queue (Celery + Redis or AWS SQS) so retrieval happens synchronously while generation is async. Return the answer via websocket or polling.
- Horizontal scaling: run multiple Falcon inference replicas behind a load balancer. Use GPU-accelerated inference servers (TGI, vLLM, or TorchServe) that support batching and streaming.
- FAISS index: move to a persistent vector store (Pinecone, Weaviate, or pgvector) that supports concurrent reads and index updates without downtime.

**Target capacity:** 1,000 concurrent users requires approximately 10–20 GPU inference replicas depending on average response latency and request rate.

### Infrastructure (MLOps Pipeline)

```
PDF ingestion → chunking → embedding → vector store
                                              ↓
User question → retrieval → Falcon generation → response
                                              ↓
                                         evaluation metrics logged
```

**Components needed:**
- **Data pipeline:** Automated PDF processing triggered when new FAA handbooks are released. Validates chunk quality before indexing.
- **Model registry:** Track versions of the embedding model, Falcon checkpoint, and system prompt. Tag each deployment.
- **Vector store versioning:** Keep previous index versions for rollback. New index is built and validated before replacing the production one.
- **CI/CD:** On every code change, run the evaluation suite against the production dataset. Block deployment if Retrieval < 0.55, Relevance < 0.75, or Faithfulness < 0.90.
- **Infrastructure as code:** Terraform or CDK for all cloud resources. No manual configuration.

### Monitoring & Observability

**What to track:**
- Per-request latency (P50, P95, P99) for retrieval and generation separately
- Retrieval score distribution — a drop indicates embedding drift or index degradation
- Faithfulness score over time — a drop indicates prompt drift or model behavior change
- Out-of-domain detection rate — how often the system correctly rejects irrelevant questions
- User satisfaction (thumbs up/down on responses)

**Model drift detection:**
- Run the evaluation dataset weekly against production. Alert if any metric drops more than 5% week-over-week.
- Track embedding distribution shifts using cosine similarity between weekly samples.
- If the FAA publishes new regulations, trigger a full re-evaluation after re-indexing.

**Tooling:** Prometheus + Grafana for metrics. Structured logging to Elasticsearch. Evaluation results stored in a time-series database for trend analysis.

### Privacy & Safety

**Student data:**
- Do not store conversation history server-side unless the student explicitly opts in.
- If stored, encrypt at rest (AES-256) and in transit (TLS 1.3).
- Implement retention policies — delete conversation logs after 90 days.
- Comply with FERPA (for US universities) — student learning data is protected.

**Content guardrails:**
- Input validation: reject questions over 500 tokens to prevent prompt injection.
- Output filtering: scan responses for PII before returning to the student.
- The current out-of-domain detection (score threshold) acts as a topic guardrail. Extend it to also reject questions that are aviation-related but outside the scope of the specific curriculum.
- Rate limiting per student to prevent abuse.

**Model safety:**
- Falcon is an open-weight model — no data leaves the infrastructure.
- Audit logs for all queries and responses, accessible only to instructors and administrators.
- Regular red-teaming to test for prompt injection, jailbreaks, and off-topic responses.

### Deployment Strategy

**Environments:** development → staging → production. Staging mirrors production infrastructure at 10% scale.

**Deployment process:**
1. Build new Docker image with updated code or model
2. Run evaluation suite in staging against the standard dataset
3. Run canary deployment: 5% of traffic to new version, 95% to current
4. Monitor metrics for 24 hours
5. If no regression, gradually shift 100% of traffic
6. Keep previous version running for 48 hours for instant rollback

**Rollback trigger:** Any metric dropping more than 10% from baseline triggers automatic rollback.

**Zero-downtime updates:** Rolling deployments with health checks. New replicas must pass a health check (successfully answer 3 test questions) before receiving traffic.

### Usage Control

**Preventing misuse:**
- Authentication required — integration with the university's SSO (OAuth2/SAML).
- Per-student rate limits: 50 questions per day for normal use, 200 for exam preparation periods.
- Instructor override: instructors can query the system without limits for curriculum development.
- Abuse detection: flag accounts asking repeated out-of-domain questions or attempting prompt injection.

**Educational value preservation:**
- The system should assist learning, not replace it. Consider adding a "hint mode" that provides partial information and prompts the student to think, rather than always giving complete answers.
- Instructors can configure which chapters of the FAA handbook are active per course stage — a student in early training should not receive information about advanced maneuvers.
- Conversation summaries sent to instructors (with student consent) to identify common misunderstandings.

**Academic integrity:**
- Clearly label all responses as AI-generated.
- Log which questions were asked before exams to detect misuse patterns.
- Do not generate answers to practice exam questions if the system can identify them as such.

---

## Summary

| Area | Status |
|---|---|
| Evaluation framework (3 metrics) | Implemented |
| Retrieval accuracy | Implemented + fixed |
| Answer relevance (NLI) | Implemented |
| Faithfulness (LLM judge) | Implemented |
| Bug: single chunk context | Fixed |
| Bug: no score threshold | Fixed |
| Bug: character-based chunking | Fixed |
| Bug: dead MCP tool | Fixed |
| Bug: follow-up question memory | Fixed |
| Prompt engineering | Implemented |
| Semantic chunking | Implemented |
| MCP chat on Windows | Partial (workaround via direct-chat) |
| Production roadmap | Documented above |
