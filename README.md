---
title: Multi-Channel Retail Environment
emoji: 🏪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
pinned: false
---

# 🏪 Multi-Channel Retail Environment with Disruption Recovery

> **OpenEnv Scaler Hackathon Submission** — A production-grade, OpenEnv-compliant environment that challenges AI agents to manage a realistic multi-channel retail operation under supply chain disruptions, demand shocks, and stochastic supplier behavior.

[![Live on HF Spaces](https://img.shields.io/badge/🤗-Live%20on%20Spaces-blue)](https://huggingface.co/spaces/RohanExploit/Meta-hackathon)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-green)]()
[![Tests](https://img.shields.io/badge/tests-6%2F6%20pass-brightgreen)]()

---

## 🎯 What Problem Does This Solve?

Retail operations managers face a daily challenge that no toy environment captures: **simultaneously optimizing pricing, inventory, and promotions across multiple customer segments while dealing with unpredictable supply chain disruptions.**

This environment faithfully simulates the real-world tradeoffs:

- A **demand collapse** hits your bestseller — do you slash prices or run a promotion?  
- Your supplier delivers only **80% of orders** — how much buffer stock do you carry?  
- Luxury customers will pay 2× but buy less — how do you **allocate limited inventory** between segments?  
- Holding costs eat into margins — when exactly should you reorder?  

These are genuine decisions that cost retailers billions annually. This is not a game.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server (server/app.py)           │
│  POST /reset · POST /step · GET /state · POST /evaluate     │
│  GET /tasks · GET /health · POST /live/start · /stop · ...   │
├─────────────────────────────────────────────────────────────┤
│                     Core Environment                         │
│  environment/retail_env.py   ← MultiChannelRetailEnv         │
│  environment/models.py       ← Pydantic: Action, Obs, State │
│  environment/grader.py       ← Deterministic scoring [0,1]  │
│  environment/tasks.py        ← 5 difficulty tiers            │
├─────────────────────────────────────────────────────────────┤
│              Baseline Agent (inference.py)                    │
│  OpenAI-compatible client → observe → reason → act loop      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🕹️ Action Space

The agent selects **one action per step** from five typed Pydantic models:

| Action | JSON Schema | Purpose |
|--------|------------|---------|
| **Allocate** | `{"action": "allocate", "product": "str", "luxury_units": int, "budget_units": int}` | Assign inventory between luxury (high-margin) and budget (high-volume) segments |
| **Set Price** | `{"action": "set_price", "product": "str", "segment": "luxury\|budget", "new_price": float}` | Dynamic pricing within bounds — higher price reduces demand but increases margin |
| **Order** | `{"action": "order", "product": "str", "quantity": int}` | Purchase from supplier — cost deducted immediately, delivery has stochastic lead time |
| **Promote** | `{"action": "promote", "product": "str", "budget_allocated": float}` | Run a promotional campaign — upfront cash cost to stimulate demand |
| **NoOp** | `{"action": "noop"}` | Hold current strategy — wait and observe |

---

## 👁️ Observation Space

The agent receives a **partial observation** (true demand patterns are hidden):

```python
RetailObservation(
    day: int,                    # Current day [0, horizon)
    cash: float,                 # Available funds
    inventory: Dict[str, int],   # Current stock by product
    recent_demand_luxury: Dict,  # Demand signal (luxury segment)
    recent_demand_budget: Dict,  # Demand signal (budget segment)
    recent_stockouts: Dict,      # Stockout indicators per product
    prices_luxury: Dict,         # Current luxury prices
    prices_budget: Dict,         # Current budget prices
    disruption_active: bool,     # Is a disruption currently happening?
    disruption_severity: float,  # Severity [0.0–1.0]
    market_confidence: float,    # Decaying confidence indicator [0.0–1.0]
)
```

The agent must infer hidden demand patterns from noisy signals — this is a partially observable environment by design.

---

## 📊 Tasks (5 Difficulty Tiers)

| Task | Horizon | Products | Supplier Reliability | Key Challenge | Baseline Profit |
|------|---------|----------|---------------------|---------------|-----------------|
| `easy` | 10 days | 1 | 100% | Stable demand, no disruptions | $50 |
| `medium_simple` | 14 days | 2 | 95% | Variable demand, occasional disruptions | $100 |
| `medium_challenge` | 14 days | 2 | 88% | Supply delays, demand elasticity shifts | $80 |
| `hard` | 21 days | 3 | 85% | Frequent disruptions, tight margins | $120 |
| `expert` | 30 days | 4 | 80% | Extreme chaos: multi-product, high variance | $200 |

Each task has a **fixed seed** for reproducible evaluation.

---

## 🏆 Grading System

The grader (`environment/grader.py`) is **deterministic, transparent, and stateless**. It computes a composite score in `[0.0, 1.0]`:

```
score = 0.5 × profit_score + 0.3 × fill_rate_score + 0.2 × efficiency_score
```

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| **Profit Score** | `clamp(profit / baseline_profit, 0, 1)` | 50% | Primary business metric |
| **Fill Rate Score** | `clamp(total_sales / total_demand, 0, 1)` | 30% | Customer satisfaction proxy |
| **Efficiency Score** | `clamp(1 - holding_cost / max_possible_holding_cost, 0, 1)` | 20% | Capital efficiency |

### Anti-Gaming Guardrail

If `fill_rate < 0.6`, the final score is **halved**. This prevents agents from exploiting high-margin strategies that ignore customer demand.

### Multi-Seed Aggregation

The `/evaluate` endpoint supports multi-seed evaluation with optional variance penalty:

```
adjusted_score = mean_score - variance_penalty × variance(seed_scores)
```

---

## 🚀 Installation & Quick Start

### Prerequisites
- **Python 3.10+** (if running locally)
- **Docker** (optional, if running via containerized setup)
- **Hugging Face Token / API Key** (to run the baseline inference agent)

### Local Installation & Development

```bash
# Clone the repository
git clone https://github.com/RohanExploit/Meta-hackathon.git
cd Meta-hackathon

# Setup
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -e .

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Baseline Inference

In a second terminal:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=NousResearch/Nous-Hermes-2-Mistral-7B-DPO
export OPENAI_API_KEY=<your-hf-token>

python inference.py
```

### Docker

```bash
docker build -t retail-env .

docker run -p 8000:8000 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="NousResearch/Nous-Hermes-2-Mistral-7B-DPO" \
  -e OPENAI_API_KEY="<token>" \
  retail-env
```

### Hugging Face Spaces

This repository is deployed live at **[huggingface.co/spaces/RohanExploit/Meta-hackathon](https://huggingface.co/spaces/RohanExploit/Meta-hackathon)**. The Space auto-builds from the root `Dockerfile` and starts serving the API immediately.

---

## 📡 API Reference

All endpoints are served by FastAPI with automatic OpenAPI docs at `/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/tasks` | GET | List all 5 tasks with metadata |
| `/reset` | POST | Reset environment to a task: `{"task_name": "easy", "seed": 42}` |
| `/step` | POST | Execute action: `{"action": {"action": "order", "product": "Product_A", "quantity": 5}}` |
| `/state` | GET | Full internal state (for debugging) |
| `/evaluate` | POST | Score an episode summary or batch of summaries |
| `/live/start` | POST | Start real-time continuous stepping (heuristic or noop mode) |
| `/live/stop` | POST | Stop real-time stepping |
| `/live/status` | GET | Real-time runner status |
| `/live/latest` | GET | Latest real-time tick observation |

### Example: Full Episode

```bash
# 1. Reset to easy task
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy", "seed": 42}'

# 2. Take an action
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "order", "product": "Product_A", "quantity": 3}}'

# 3. Evaluate
curl -s -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"summary": {"profit": 80, "baseline_profit": 100, "total_demand": 100, "total_sales": 75}}'
```

---

## 🧪 Testing & Validation

### Compliance Tests (6 tests)

```bash
python -m pytest tests/test_compliance.py -v
```

Validates: OpenEnv YAML contract, HF Spaces config, task count, API endpoints, grader bounds, anti-gaming behavior.

### Smoke Tests

```bash
python test_env.py    # Environment lifecycle
python test_api.py    # API simulation
```

### Health Check (all-in-one)

```bash
python scripts/health_check.py
```

Runs compile checks → test suite → smoke scripts → pre-submission checks.

### Pre-Submission Check

```bash
python scripts/pre_submission_check.py
```

Validates: task count ≥ 3, inference env vars, Docker availability, OpenEnv CLI.

### Reproducibility Benchmark

```bash
# Requires running server + API credentials
python scripts/benchmark_inference.py
```

Runs inference twice, compares scores for determinism.

---

## 📂 Repository Structure

```
Meta-hackathon/
├── Dockerfile                  # Root-level Docker config (HF Spaces uses this)
├── README.md                   # This file
├── app.spaces.yaml             # HF Spaces auto-config (sdk: docker, tag: openenv)
├── openenv.yaml                # OpenEnv environment metadata
├── pyproject.toml              # Dependencies + entry points
├── uv.lock                     # Locked dependency versions
├── inference.py                # Baseline LLM agent (OpenAI-compatible)
│
├── environment/                # Core environment package
│   ├── __init__.py
│   ├── models.py               # Pydantic: RetailAction, RetailObservation, RetailState
│   ├── retail_env.py           # MultiChannelRetailEnv (reset, step, state)
│   ├── grader.py               # Deterministic scoring [0, 1]
│   └── tasks.py                # 5 task definitions (easy → expert)
│
├── server/                     # FastAPI server package
│   ├── __init__.py
│   ├── app.py                  # 10 API endpoints + LiveRunner
│   └── ui/
│       └── index.html          # Terminal-style web UI
│
├── tests/
│   └── test_compliance.py      # 6 compliance tests
│
├── scripts/
│   ├── health_check.py         # All-in-one validation
│   ├── pre_submission_check.py # Rubric-mapped checks
│   ├── benchmark_inference.py  # Reproducibility benchmark
│   └── run_tests.py            # Isolated test runner
│
├── test_env.py                 # Environment smoke test
├── test_api.py                 # API smoke test
└── .github/workflows/ci.yml   # GitHub Actions CI
```

---

## 🔑 Key Design Decisions

### Why Multi-Segment Pricing?
Real retailers optimize per customer tier. Luxury customers accept premium prices with lower volume; budget customers demand high volume at thin margins. This creates a genuine tradeoff that doesn't exist in single-price environments.

### Why Disruptions?
Supply chain disruptions (demand collapses, supply delays, demand spikes) occur with ~15% probability per step. They test whether agents can **adapt and recover** — a capability critical in real-world retail operations.

### Why Transparent Grading?
The grader formula is fully documented and deterministic. Agents can learn that profit matters most (50% weight), customer satisfaction is essential (30%), and capital efficiency rounds out performance (20%). No hidden weights.

### Why Partial Observability?
Agents see demand signals, not true demand distributions. This mirrors reality: retailers observe sales data and stockout counts, not the underlying customer arrival process.

---

## 📈 Expected Baseline Scores

| Task | Heuristic Agent | LLM Agent (7B) | Optimal Estimate |
|------|----------------|-----------------|------------------|
| easy | ~0.85 | ~0.90 | ~0.95 |
| medium_simple | ~0.55 | ~0.70 | ~0.85 |
| medium_challenge | ~0.45 | ~0.60 | ~0.80 |
| hard | ~0.30 | ~0.45 | ~0.70 |
| expert | ~0.20 | ~0.35 | ~0.60 |

---

## 🔄 Reproducibility

- All randomness is **seeded** (`np.random.seed`, `random.seed`) via task config
- Inference uses `temperature=0` for deterministic LLM output
- Each task has a **fixed grading seed** for reproducible evaluation
- Episode metrics are fully logged in the terminal step info
- Docker build is deterministic with pinned dependencies (`uv.lock`)

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | For inference | LLM API endpoint (e.g., `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | For inference | Model identifier (e.g., `NousResearch/Nous-Hermes-2-Mistral-7B-DPO`) |
| `OPENAI_API_KEY` | For inference | API key / HF token for authentication |
| `PORT` | Optional | Server port (default: `8000`) |

---

## 📋 Hackathon Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Real-world task simulation | ✅ | Multi-channel retail with disruptions |
| Full OpenEnv spec (typed models, step/reset/state, openenv.yaml) | ✅ | Pydantic models + 10 API endpoints |
| Minimum 3 tasks (easy → hard, scores 0.0–1.0) | ✅ | 5 tasks with graded difficulty |
| Meaningful reward function with partial progress signals | ✅ | Per-step: revenue − stockout penalty − holding costs |
| Baseline inference script with reproducible scores | ✅ | `inference.py` with seeded episodes |
| Deploy to HF Spaces + working Dockerfile | ✅ | [Live Space](https://huggingface.co/spaces/RohanExploit/Meta-hackathon) |
| README with environment description, spaces, setup | ✅ | This document |
| Agent graders (scores 0.0–1.0) | ✅ | `environment/grader.py` — deterministic, bounded |
| CI/CD | ✅ | `.github/workflows/ci.yml` |

---

## 📜 License

MIT

---

*Built for the Meta OpenEnv Scaler Hackathon 2026.*
