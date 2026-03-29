# Multi-Channel Retail Environment with Disruption Recovery

An OpenEnv-compliant environment for training AI agents in dynamic retail management with multi-segment pricing, supply chain disruptions, and real-time adaptation.

## Overview

This environment simulates a realistic retail operation where an AI agent manages:
- **Multi-channel customer segments** (luxury vs. budget with different price sensitivities)
- **Dynamic demand shocks** (disruptions, seasonality, price elasticity)
- **Supply chain complexity** (stochastic lead times, unreliable suppliers)
- **Real-time optimization** (inventory allocation, pricing, promotions, ordering)

This is a genuine business problem that retailers solve daily, not a toy optimization task.

## Innovation

Unlike generic inventory environments, this system includes:
1. **Demand disruptions** (20% chance per step of collapse/spike/delay) that test recovery strategies
2. **Segment-specific pricing** enabling dynamic margin optimization
3. **Supplier unreliability** (80-95% fulfillment rate) requiring buffer stock planning
4. **Transparent profit-based grading** (Score = Profit/Baseline, no hidden formulas)
5. **Five graded tasks** from trivial (easy) to very hard (4-product, 30-day, high chaos)

## Action Space

### 1. **Allocate**
```json
{"action": "allocate", "product": "str", "luxury_units": int, "budget_units": int}
```
- Assign inventory between luxury (high-margin) and budget (high-volume) segments
- Enables dynamic channel strategy

### 2. **Set Price**
```json
{"action": "set_price", "product": "str", "segment": "luxury|budget", "new_price": float}
```
- Set price for a segment (within price bounds)
- Higher price → lower demand but higher margin
- Subject to demand elasticity

### 3. **Order**
```json
{"action": "order", "product": "str", "quantity": int}
```
- Place order with stochastic lead time (mean 2-3 days, variance 1-3)
- Cost deducted immediately
- Supplier success rate: 80-95%

### 4. **Promote**
```json
{"action": "promote", "product": "str", "budget_allocated": float}
```
- Run promotional campaign (budget_allocated × 1.5-2.0x demand multiplier)
- Upfront cash cost

### 5. **NoOp**
```json
{"action": "noop"}
```
- Do nothing (hold current strategy)

## Observation Space

The agent receives (partial state, hidden from true values):
```python
{
  "day": int,                              # Current day (0-30)
  "cash": float,                           # Available funds
  "inventory": {"Product_A": 10, ...},    # Current stock by product
  "recent_demand_luxury": {...},          # Noisy estimate of luxury demand
  "recent_demand_budget": {...},          # Noisy estimate of budget demand
  "recent_stockouts": {...},              # Count of stockouts per product
  "prices_luxury": {"Product_A": 12.0},  # Current luxury prices
  "prices_budget": {"Product_A": 8.0},   # Current budget prices
  "disruption_active": bool,              # Is disruption ongoing?
  "disruption_severity": 0.3-0.9,         # Magnitude of disruption
  "market_confidence": 0.6-0.9            # Agent's uncertainty (decreases over episode)
}
```

## Reward Structure

Step-wise reward = Revenue - Order Costs - Holding Costs - Stockout Penalties ± Disruption Effects

This aligns with the final profit-based score.

## Tasks (5 levels)

| Task | Horizon | Products | Complexity | Baseline Profit |
|------|---------|----------|-----------|-----------------|
| easy | 10 days | 1 | Stable demand, no disruptions | $50 |
| medium_simple | 14 days | 2 | Variable demand, rare disruptions | $100 |
| medium_challenge | 14 days | 2 | Supply delays, demand shifts | $80 |
| hard | 21 days | 3 | Frequent disruptions, 85% supplier reliability | $120 |
| expert | 30 days | 4 | Chaos: extreme disruptions, tight margins | $200 |

## Grading (Transparent)

**Primary Score:**
```
score = (profit / baseline_profit)  clamped to [0, 1]
```

**Guardrails:**
- If fill_rate < 0.5: score *= 0.5
- If fill_rate < 0.7: score *= 0.85
- Disruption recovery bonus: +0.15 for 100% recovery rate

**Example:**
- Profit = $100, Baseline = $100, Fill Rate = 80% → Score = 1.0
- Profit = $50, Baseline = $100, Fill Rate = 60% → Score = 0.5 × 0.85 = 0.425

## Setup & Usage

### Local Installation

```bash
# Clone repo
git clone <repo-url>
cd retail_inventory_env

# Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install
pip install --upgrade pip
pip install -e .

# Set API credentials
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=<your-model>
export HF_TOKEN=<your-token>

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
python inference.py
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t retail-env .

# Run
docker run -p 8000:8000 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="<model>" \
  -e HF_TOKEN="<token>" \
  retail-env
```

Note: The Dockerfile now copies the full repo before `pip install -e .` so editable
install works reliably during image builds.

### Hugging Face Spaces

Deploy this repo to HF Spaces; the space will automatically serve the API.

## API Examples

### Reset to a task
```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy", "seed": 42}'
```

### Take a step
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "order", "product": "Product_A", "quantity": 5}}'
```

### List tasks
```bash
curl http://localhost:8000/tasks
```

### Evaluate episode
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"summary": {"profit": 100, "baseline_profit": 100, "total_demand": 50, "total_sales": 45}}'
```

## Files

- `environment/models.py` — Pydantic models (Action, Observation, State)
- `environment/retail_env.py` — Core environment (MultiChannelRetailEnv)
- `environment/grader.py` — Transparent profit-based grader
- `environment/tasks.py` — 5 tasks with configs
- `server/app.py` — FastAPI server (OpenEnv compliant)
- `server/Dockerfile` — Containerization
- `inference.py` — Baseline agent using OpenAI client
- `openenv.yaml` — Environment metadata
- `pyproject.toml` — Dependencies

## Key Design Decisions

### Why Multi-Segment?
Real retailers optimize per customer tier. Luxury customers accept premium prices; budget-conscious demand high volume. This creates a non-trivial tradeoff.

### Why Disruptions?
Disruptions (demand collapses, supply delays) are common IRL and test agent adaptability. Recovery under uncertainty is a real challenge.

### Why Transparent Grader?
Previous rubric-based graders with many weights confused agents. Profit-based is interpretable: agents learn it's profit that matters.

### Why 5 Tasks?
Progression from trivial (1 product, stable) to expert (4 products, chaos) ensures agents generalize across complexity levels.

## Reproducibility

- All randomness is seeded (np.random.seed, random.seed)
- inference.py uses temperature=0 for determinism
- Each task has fixed seeds for reproducible grading
- Episode metrics are fully logged

## Performance Baselines

Expected scores:
- **Easy**: ~0.9 (trivial; any reasonable agent wins)
- **Medium Simple**: ~0.7 (basic profitability)
- **Medium Challenge**: ~0.6 (supply/demand adaptation)
- **Hard**: ~0.4-0.5 (multi-product chaos; requires planning)
- **Expert**: ~0.3-0.4 (extreme case; only sophisticated strategies survive)

## Validation

```bash
# Check syntax
python -m py_compile environment/*.py server/app.py inference.py

# Run tests
python -m pytest tests/ -v

# Build Docker
docker build -f server/Dockerfile -t retail-env .

# Validate openenv.yaml
# (Requires openenv CLI tool)
openenv validate openenv.yaml
```

### Stable local test runner (recommended on this machine)

The host can have globally installed pytest plugins that conflict with this repo.
Use the wrapper below to disable host plugin autoload safely:

```bash
python scripts/run_tests.py
```

### One-command system health check

Run compile checks + tests + smoke scripts:

```bash
python scripts/health_check.py
```

### Pre-submission gate (rubric-mapped)

Runs health checks, task-count checks, inference env-var checks, and optional
Docker/OpenEnv CLI checks:

```bash
python scripts/pre_submission_check.py
```

Useful options:

```bash
# Skip Docker check (if Docker not installed on local machine)
set PRECHECK_SKIP_DOCKER=1

# Skip OpenEnv CLI check (if openenv CLI not installed)
set PRECHECK_SKIP_OPENENV=1
```

### CI

GitHub Actions workflow is provided at `.github/workflows/ci.yml` and sets
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid plugin pollution in CI as well.

## Notes

- **Time Complexity**: inference.py runs all 5 tasks in <20 min on 2 vCPU / 8GB
- **Dependencies**: FastAPI, Pydantic, NumPy, OpenAI client
- **Compatibility**: Python 3.10+
