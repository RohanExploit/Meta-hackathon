# Retail Inventory Management Environment

An OpenEnv-compliant environment for training AI agents in retail inventory and pricing management.

## Environment Description

This environment simulates a retail store where an AI agent must manage inventory levels and pricing strategies to maximize profit over time. The agent makes decisions about ordering stock and setting prices while facing random demand patterns, holding costs, and potential stockouts.

## Action Space

The agent can take one of three actions at each step:

1. **Order**: `{"action": "order", "product": "product_name", "quantity": integer}`
   - Order more units of a product
   - Quantity must be positive integer
   - Limited by available cash and storage capacity

2. **Set Price**: `{"action": "set_price", "product": "product_name", "new_price": float}`
   - Set the selling price for a product
   - Price must be between cost+10% and cost+200% of product cost

3. **NoOp**: `{"action": "noop"}`
   - Take no action this step

## Observation Space

At each step, the agent receives an observation containing:

- `inventory`: Dict mapping product names to current stock levels
- `cash`: Current available funds (float)
- `day`: Current day in the episode (integer)
- `sales_history`: Dict mapping product names to units sold previous day
- `price`: Dict mapping product names to current selling prices

## Reward Function

The agent receives a reward at each step that reflects immediate profit contribution:

```
reward = (today's sales revenue) - (order cost) - (holding cost) - (stockout penalty)
```

Where:
- Sales revenue = units sold × selling price
- Order cost = units ordered × unit cost (if ordering)
- Holding cost = units in inventory × holding cost rate per unit
- Stockout penalty = penalty for unmet demand (encourages adequate stocking)

This reward shaping aligns with the final grading metric (total profit) while providing intermediate feedback.

## Task Design

The environment includes three progressively challenging tasks:

### Easy Task: Single Product Steady Demand
- **Objective**: Maximize profit for one product over 7 days
- **Products**: 1 product with fixed daily demand
- **Initial State**: $100 cash, 10 units inventory
- **Constraints**: Unlimited inventory capacity, no lead time
- **Optimal Strategy**: Order exactly to meet demand each day

### Medium Task: Two Products, Variable Demand
- **Objective**: Manage two products over 14 days to maximize profit
- **Products**: 2 products with different costs and random demand
- **Initial State**: $200 cash, moderate inventory of each
- **Constraints**: Limited inventory capacity, lead time for orders, order fees
- **Challenge**: Balance order costs vs stockouts with stochastic demand

### Hard Task: Multi-product, Time-varying Demand & Pricing
- **Objective**: Manage 3 products over 30 days with changing demand patterns
- **Products**: 3 products with distinct characteristics
- **Initial State**: $300 cash, low initial inventory
- **Constraints**: Tight inventory limits, dynamic pricing restrictions
- **Challenge**: Detect trends and adapt strategies across multiple products

## Grading

Performance is scored using normalized profit:

```
Score = min(max(Profit / TargetProfit, 0.0), 1.0)
```

Where:
- Profit = final_cash + inventory_value - initial_cash
- TargetProfit = precomputed optimal profit for the task
- Score ranges from 0.0 (loss) to 1.0 (meeting/exceeding target)

This provides continuous, non-binary evaluation that rewards partial progress.

## Setup and Usage

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker

### Installation Guide (Local)

1. Clone the repository and enter the project:
   ```bash
   git clone <your-repo-url>
   cd retail_inventory_env
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   On Linux/macOS:
   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. (Optional) Set inference environment variables:
   ```bash
   set API_BASE_URL=https://router.huggingface.co/v1
   set MODEL_NAME=<your-model-name>
   set HF_TOKEN=<your-token>
   ```
   On Linux/macOS:
   ```bash
   export API_BASE_URL=https://router.huggingface.co/v1
   export MODEL_NAME=<your-model-name>
   export HF_TOKEN=<your-token>
   ```

5. Run quick tests:
   ```bash
   python test_env.py
   python test_api.py
   ```

6. Start the API server:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

7. Run inference script:
   ```bash
   python inference.py
   ```

### Docker Deployment

1. Build the container:
   ```bash
   docker build -f server/Dockerfile -t retail-inventory-env .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 \
     -e API_BASE_URL="your_api_base_url" \
     -e MODEL_NAME="your_model_name" \
     -e HF_TOKEN="your_hf_token" \
     retail-inventory-env
   ```

### Hugging Face Spaces

The environment is configured for deployment to Hugging Face Spaces:
1. Push this repository to HF Spaces
2. Set the required secrets (API_BASE_URL, MODEL_NAME, HF_TOKEN)
3. The space will automatically serve the environment API

## API Quickstart

Reset:
```bash
curl -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d "{\"task_config\":{\"products\":[\"Widget\"],\"initial_inventory\":{\"Widget\":10},\"initial_cash\":100.0,\"fixed_demand\":{\"Widget\":2.0},\"product_costs\":{\"Widget\":5.0},\"holding_costs\":{\"Widget\":0.1},\"max_inventory\":{\"Widget\":20},\"horizon\":7,\"baseline_profit\":30.0},\"seed\":42}"
```

Step:
```bash
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d "{\"action\":{\"action\":\"order\",\"product\":\"Widget\",\"quantity\":2}}"
```

## Validation

Run local quality checks:
```bash
python -m py_compile environment/models.py environment/retail_env.py server/app.py inference.py
python test_env.py
python test_api.py
```

## Files

- `environment/retail_env.py`: Core environment implementation
- `environment/models.py`: Pydantic models for state, observation, actions
- `server/app.py`: FastAPI server exposing environment endpoints
- `server/Dockerfile`: Container definition for deployment
- `inference.py`: Baseline agent script showing interaction pattern
- `openenv.yaml`: Environment metadata specification
- `pyproject.toml`: Dependencies and package configuration

## Implementation Notes

- The environment uses seeded randomness for reproducibility
- All monetary values are in dollars
- Time steps represent days
- The agent only observes partial state (no access to true demand patterns)
- Actions are validated and invalid actions receive small penalties
