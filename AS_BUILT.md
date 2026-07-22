# As-Built Planning Document

## Project title

**Resonance Ticket Triage Agent (LangGraph Supervisor + HITL)**

Package: `RTTA-AI-Multi-Agent-Ticket-Triage` v0.1.0 — Resonance Technologies Agentic AI Lab **Project 2**.  
Sibling Project 1 (Research Assistant) lives in `../RAIRA-AI-Research-Assistant/` and shares `.env` by default.

---

## Purpose

A multi-agent ticket triage system that:

1. Classifies tickets (category + severity)  
2. Investigates with domain tools (logs, metrics, runbooks, history)  
3. Drafts a customer reply using procedural / episodic / semantic memory  
4. Pauses for human approval (HITL) before send  
5. Dispatches the approved reply (mock log), with Slack notify on P1  

UI: FastAPI approval console at `http://127.0.0.1:8002`.

---

## Locked decisions

| Area | Decision (as built) |
|------|---------------------|
| Orchestration | LangGraph **supervisor loop** — workers never route to each other |
| Workers | `triager` → `investigator` → `responder` → `hitl` → `send` |
| Chat model (default) | `bedrock_converse:openai.gpt-oss-120b-1:0` via `RTTA_MODEL` |
| Vertex swap | `google_vertexai:gemini-2.5-pro` (+ `GCP_PROJECT` / `GCP_LOCATION`) |
| Offline / throttle | `RTTA_MODEL=fake`; auto fake fallback on Bedrock quota (configurable) |
| Embeddings | `bedrock:amazon.titan-embed-text-v2:0` (or fake hashed vectors) |
| Local checkpointer | `SqliteSaver` → `checkpoints.sqlite` |
| AgentCore checkpointer | `PostgresSaver` via `POSTGRES_DSN` |
| Semantic memory | LangGraph Store (`RTTA_MEMORY=memory` or `postgres`) |
| Episodic memory | pgvector `past_resolutions` + `data/{domain}/historical_tickets.jsonl` fallback |
| Procedural memory | Versioned prompts in `data/prompts/responder_{domain}.json` (`latest`) |
| HITL | `langgraph.types.interrupt` inside `hitl_node` only |
| Investigator tools | Mocks only — never real ops APIs in class |
| Tool budget | Max **8** investigator tool calls |
| Outbound send | Mock append to `data/sent_responses.log` |
| P1 Slack | `notify_slack("#incidents", blocks)` after approved P1 send |
| Input guardrail | Hard-block injection patterns in `app/guardrails.py` (+ optional Bedrock Guardrail) |
| Runbooks | `RTTA_RUNBOOKS=auto` → Bedrock KB if `BEDROCK_KB_ID` set, else files |
| Deploy | Bedrock AgentCore **and** Vertex AI Agent Engine |
| AgentCore managed memory | Disabled by default (`DISABLE_AGENTCORE_MEMORY=1`) |
| Evals | LangSmith experiments; security pass bar **≥ 95%** (19/20) |
| UI stack | FastAPI + fetch-polled approval page — no React/npm |
| Vertex Model Armor | **Out of scope** (GCP-side) |

---

## Architecture

### Graph flow (as built)

```
START → supervisor
          ├─ triager       → supervisor
          ├─ investigator  → supervisor
          ├─ responder     → supervisor
          ├─ hitl          → supervisor   (interrupt until approve/edit/reject)
          ├─ send          → supervisor
          └─ END
```

| Route condition | Next |
|-----------------|------|
| `approval == "rejected"` | `END` (guardrail / operator reject) |
| `classification is None` | `triager` |
| `findings == []` | `investigator` |
| `draft is None` | `responder` |
| `approval == "pending"` | `hitl` |
| approved/edited and not `sent` | `send` |
| else | `END` |

### State (`TicketState`)

`ticket_id`, `raw`, `domain` (`support` \| `it-helpdesk` \| `oncall`),  
`classification`, `severity` (`P1`–`P4`), `findings`, `draft`,  
`approval` (`pending` \| `approved` \| `edited` \| `rejected`),  
`sent`, `step_log`, `next`

### Memory layers (Responder)

| Layer | Module | Behaviour |
|-------|--------|-----------|
| Procedural | `app/memory/procedural.py` | Style prompt from disk (`latest`) |
| Episodic | `app/memory/episodic.py` | Similar past resolutions (pgvector or JSONL) |
| Semantic | `app/memory/semantic.py` | Per-user facts via Store (`recall_user`) |

### Investigator tools

| Tool | Source |
|------|--------|
| `query_logs` | `data/{domain}/mock_logs.json` |
| `query_metrics` | `data/{domain}/mock_metrics.json` |
| `search_runbooks` | Bedrock KB **or** `data/{domain}/runbooks/*.md` |
| `get_ticket_history` | mock / historical tickets |

### Security model

- **Blocked:** hard-block patterns or Bedrock/Vertex guardrail refusal → `approval=rejected`, no draft  
- **Escalated:** responder `recommended_action=escalate` + HITL interrupt  
- Eval: `security/injection_eval.py` vs `security/attacks.jsonl` (20 attacks)

```mermaid
flowchart TD
  START --> supervisor
  supervisor -->|classify| triager
  supervisor -->|investigate| investigator
  supervisor -->|draft| responder
  supervisor -->|approve| hitl
  supervisor -->|dispatch| send
  supervisor -->|done / rejected| END
  triager --> supervisor
  investigator --> supervisor
  responder --> supervisor
  hitl --> supervisor
  send --> supervisor
```

---

## Project layout (as built)

```
RTTA-AI-Multi-Agent-Ticket-Triage/
├── app/
│   ├── main.py                 # FastAPI approval UI (:8002)
│   ├── graph.py                # build_graph / build_graph_with_backends
│   ├── state.py
│   ├── llm.py / _fake_llm.py / guardrails.py
│   ├── hitl.py / hitl_log.py
│   ├── agents/                 # supervisor, triager, investigator, responder, hitl, send
│   ├── memory/                 # semantic, episodic, procedural
│   ├── tools/                  # investigator tools + send_response + notify_slack
│   ├── ui/                     # approval.html, styles.css
│   ├── ops/dashboard.py        # Streamlit ops dashboard
│   └── cron/refine_responder_prompt.py
├── evals/
│   ├── golden.jsonl            # e2e / triager labels
│   ├── investigator_golden.jsonl
│   ├── responder_golden.jsonl
│   ├── triager_eval.py / investigator_eval.py / responder_eval.py / e2e_eval.py
│   └── run_all.py
├── security/
│   ├── attacks.jsonl           # 20 injection payloads
│   └── injection_eval.py
├── deploy/
│   ├── deploy_agentcore.sh / agentcore_entrypoint.py (also root shim)
│   ├── vertex_engine_deploy.py / deploy_vertex_engine.sh
│   └── IAM / zip shims for Windows
├── data/
│   ├── support/                # taxonomy, mocks, runbooks, historical_tickets
│   ├── prompts/                # responder_support / it-helpdesk / oncall
│   ├── hitl_outcomes.jsonl
│   ├── sent_responses.log
│   └── slack_notifications.log
├── scripts/setup_billing_alerts.sh
├── agentcore_entrypoint.py
├── checkpoints.sqlite
├── pyproject.toml
└── AS_BUILT.md                 # this document
```

---

## Implementation plan (completed)

| Phase | Deliverable | Status |
|-------|-------------|--------|
| Day 5–6 | Graph skeleton, supervisor routing, sample ticket | Done |
| Day 6 H1 | Triager + taxonomy | Done |
| Day 6 H2 | Investigator + 4 mock tools | Done |
| Day 6 H3 | Three memory layers + real Responder | Done |
| Day 6 H4 | HITL + approval UI (`:8002`) | Done |
| Day 7 | AgentCore deploy + Vertex Engine deploy scripts | Done |
| Day 8 H1 | Component + e2e evals → LangSmith | Done |
| Day 8 H2 | Security injection eval (20 attacks) | Done |
| Day 8 H3 | Billing alerts script | Done |
| Stretch | Slack P1, refine cron, Bedrock KB runbooks, ops dashboard | Done |

### Known gaps (still open)

- Triager episodic few-shot examples (`# TODO` in `triager.py`)  
- `remember_user` API exists but is not written from the graph  
- Full mock corpora only under `data/support/` (`it-helpdesk` / `oncall` = prompts only)

---

## Demo ticket set (for report screenshots)

### Primary UI path

1. Start UI: `uv run python -m app.main` → open `http://127.0.0.1:8002`  
2. Click **Demo ticket** (`POST /ingest/demo`) → **TKT-1001** MFA loop  
3. Wait for HITL pending → approve / edit / reject in UI  
4. Confirm send log line in `data/sent_responses.log`

### Golden tickets (`evals/golden.jsonl`) — screenshot set

| ID | Scenario | Expected |
|----|----------|----------|
| **TKT-E01** | MFA / login loop | `login_issue`, P2, send |
| **TKT-E02** | Refund / double charge | `billing`, P2, **escalate** |
| **TKT-E03** | PDF export hang | `bug_report`, P3, send |
| **TKT-E04** | Dark mode ask | `feature_request`, P4, send |
| **TKT-E05** | Suspicious login (P1) | `account_security`, P1, **escalate** |
| **TKT-E06** | Slack webhook help | `integration_help`, P3, send |

### Security screenshots

| Attack name | Expected |
|-------------|----------|
| `ignore_previous_instructions` / `jailbreak_dan` | **blocked** |
| `refund_by_eod` / `ssn_in_ticket` / `skip_approval_send` | **escalated** |

### Ops / stretch screenshots

- Streamlit dashboard: category bars, severity, escalation rate, HITL latency, last 20 badges  
- Refine cron stdout proposing `v+1` procedural prompt (not auto-applied)  
- P1 send → `data/slack_notifications.log` (or live Slack if `SLACK_BOT_TOKEN` set)

### Screenshot checklist

- [ ] Approval UI with pending draft (classification + severity + findings)  
- [ ] Edit path changing draft body before approve  
- [ ] Escalation ticket (`TKT-E02` or `TKT-E05`) showing `recommended_action=escalate`  
- [ ] P1 Slack mock/log entry  
- [ ] Triager confusion matrix from `evals/triager_eval.py`  
- [ ] Security eval pass-rate ≥ 95%  
- [ ] Ops dashboard for “today”  
- [ ] (Optional) AgentCore / Vertex resource name from `.env.deployed`

---

## Out of scope (keeps prototype focused)

- Real Zendesk / ServiceNow / PagerDuty integrations  
- Real log/metrics APIs (Investigator stays on mocks)  
- Production email SMTP (send is a file mock)  
- Vertex Model Armor application wiring  
- Terraform / Helm / full K8s manifests  
- React / Next.js UI  
- Writing semantic memory from the live graph (`remember_user` unused)  
- Fully populated `it-helpdesk` / `oncall` mock domains  
- Auto-applying procedural prompt v+1 without instructor review  
- AgentCore managed long-term memory (explicitly disabled)

---

## Environment (key variables)

| Variable | Role / default |
|----------|----------------|
| `RTTA_MODEL` | Chat model; `fake` offline |
| `RTTA_EMBEDDINGS` | Embeddings model |
| `RTTA_MEMORY` | `memory` (default) or `postgres` |
| `RTTA_RUNBOOKS` | `auto` \| `file` \| `bedrock` |
| `POSTGRES_DSN` | `postgresql://postgres:postgres@localhost:5433/resonance` |
| `HOST` / `PORT` | `127.0.0.1` / **`8002`** |
| `BEDROCK_GUARDRAIL_ID` / `BEDROCK_GUARDRAIL_VERSION` | Optional; `DRAFT` |
| `BEDROCK_KB_ID` | Optional Knowledge Base for runbooks |
| `GCP_PROJECT` / `GCP_LOCATION` / `GCP_BUCKET` | Vertex deploy |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` | Eval upload |
| `SLACK_BOT_TOKEN` | Live Slack; else mock log |
| `ALERT_EMAIL` | Billing alerts script |
| `SECURITY_EVAL_MODEL` | Optional model override for injection eval |

`.env` resolution: `RTTA-AI-Multi-Agent-Ticket-Triage/.env` if present, else `../RAIRA-AI-Research-Assistant/.env` (`override=True`).

---

## How to run (local)

```bash
cd RTTA-AI-Multi-Agent-Ticket-Triage
uv sync

# Optional: shared Postgres for Store / episodic
# docker compose up -d postgres   # from RAIRA-AI-Research-Assistant

uv run python -m app.graph          # CLI sample ticket
uv run python -m app.main           # UI http://127.0.0.1:8002

curl -X POST http://127.0.0.1:8002/ingest/demo

# Evals / security / ops
$env:RTTA_MODEL='fake'   # PowerShell offline
uv run python evals/run_all.py
uv run python security/injection_eval.py
uv run streamlit run app/ops/dashboard.py
uv run python -m app.cron.refine_responder_prompt --domain support

# Deploy (cloud creds required)
bash deploy/deploy_agentcore.sh
bash deploy/deploy_vertex_engine.sh
ALERT_EMAIL=you@example.com bash scripts/setup_billing_alerts.sh
```

---

## Verification checklist

### Core path

- [ ] `uv sync` succeeds (Python 3.11–3.12)  
- [ ] `uv run python -m app.graph` streams supervisor → workers for `TKT-1001`  
- [ ] UI on `:8002`; `/ingest/demo` reaches `pending_hitl`  
- [ ] Approve → `sent=true` and row in `data/sent_responses.log`  
- [ ] Edit body → `approval=edited` and edited text sent  
- [ ] Reject → no send  

### Memory / responder

- [ ] Responder log shows episodic/semantic counts  
- [ ] Billing / refund ticket drafts `recommended_action=escalate`  
- [ ] P1 approved ticket appends Slack mock (`data/slack_notifications.log`)  

### Security

- [ ] Hard jailbreak ticket ends with `GUARDRAIL_REFUSAL` / blocked (no customer send)  
- [ ] `uv run python security/injection_eval.py` pass-rate ≥ 95%  

### Evals

- [ ] `evals/triager_eval.py` — category/severity + confusion matrices  
- [ ] `evals/investigator_eval.py` — keyword + judge  
- [ ] `evals/responder_eval.py` — escalation P/R + quality  
- [ ] `evals/e2e_eval.py` — category + response + escalation  
- [ ] LangSmith experiment URLs when `LANGSMITH_API_KEY` set  

### Ops / stretch

- [ ] Streamlit dashboard shows category / severity / escalation / latency  
- [ ] Refine cron prints proposed v+1 and **does not** write prompts  
- [ ] (Optional) AgentCore launch succeeds; Vertex deploy writes `.env.deployed`  

### Offline resilience

- [ ] With `RTTA_MODEL=fake` (or throttle fallback), demo + evals still complete  

---

## Evals (as built)

| Script | Measures | LangSmith prefix |
|--------|----------|------------------|
| `evals/triager_eval.py` | Category + severity exact-match; confusion matrices | `triager-eval` |
| `evals/investigator_eval.py` | Finding keyword grounding + LLM judge | `investigator-eval` |
| `evals/responder_eval.py` | Escalation precision/recall + quality 1–5 | `responder-eval` |
| `evals/e2e_eval.py` | Category + response quality + escalation (HITL auto-approved) | `e2e-eval` |
| `security/injection_eval.py` | blocked vs escalated outcomes | local printout |

Datasets: `evals/golden.jsonl`, `investigator_golden.jsonl`, `responder_golden.jsonl`, `security/attacks.jsonl`.

---

## Deploy (as built)

| Target | Entrypoint / script |
|--------|---------------------|
| Bedrock AgentCore | `agentcore_entrypoint.py` + `deploy/deploy_agentcore.sh` |
| Vertex Agent Engine | `deploy/vertex_engine_deploy.py` + `deploy/deploy_vertex_engine.sh` → `.env.deployed` |
| Billing | `scripts/setup_billing_alerts.sh` ($10/day AWS + GCP) |

`build_graph_with_backends(saver, store)` lets cloud entrypoints inject Postgres backends while local `build_graph()` uses Sqlite.

---

## As-built deltas vs early prompts

1. Approval UI uses **fetch polling** (not HTMX-only) with visible status/errors.  
2. Default app port is **8002** (avoids clash with Project 1 on 8000).  
3. AgentCore entrypoint is at **repo root** (`agentcore_entrypoint.py`) for Windows path safety.  
4. Input **guardrail** short-circuits to `approval=rejected` before triage LLM.  
5. Stretch items (Slack, refine cron, KB runbooks, ops dashboard) are **implemented**, not stubs.  
6. Semantic **writes** and triager episodic few-shots remain incomplete by design/timebox.

---

## Related docs

| Doc | Path |
|-----|------|
| Day prompts (parent) | `../project2-prompts.md` |
| Cursor rules | `../RAIRA-AI-Research-Assistant/.cursor/rules/project2-ticket-triage.mdc` |
| Shared `.env` | `../RAIRA-AI-Research-Assistant/.env` |
| Project 1 as-built | `../RAIRA-AI-Research-Assistant/AS_BUILT.md` |

---

*Document type: as-built planning record for Project 2 in `RTTA-AI-Multi-Agent-Ticket-Triage`. Update when graph topology, memory wiring, or deploy entrypoints change.*
