"""Pydantic models for the multi-channel retail environment with disruptions."""
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Allowed action types."""
    ALLOCATE = "allocate"          # Allocate inventory to customer segments
    SET_PRICE = "set_price"        # Set price for a segment
    ORDER = "order"                # Order from supplier
    PROMOTE = "promote"            # Run a promotion (budget-limited)
    NOOP = "noop"                  # Do nothing


class AllocateAction(BaseModel):
    """Allocate units of a product to luxury and budget segments."""
    action: ActionType = ActionType.ALLOCATE
    product: str
    luxury_units: int = Field(ge=0)
    budget_units: int = Field(ge=0)


class SetPriceAction(BaseModel):
    """Set price for a product in a specific segment."""
    action: ActionType = ActionType.SET_PRICE
    product: str
    segment: str  # "luxury" or "budget"
    new_price: float = Field(gt=0)


class OrderAction(BaseModel):
    """Order inventory from supplier (includes lead time risk)."""
    action: ActionType = ActionType.ORDER
    product: str
    quantity: int = Field(gt=0)
    supplier: str = "A"  # "A" for cheap/slow/unreliable, "B" for fast/reliable/expensive


class PromoteAction(BaseModel):
    """Run a promotional campaign (increases demand but costs budget)."""
    action: ActionType = ActionType.PROMOTE
    product: str
    budget_allocated: float = Field(gt=0)  # Promotional spend


class NoOpAction(BaseModel):
    """Take no action."""
    action: ActionType = ActionType.NOOP


RetailAction = Union[AllocateAction, SetPriceAction, OrderAction, PromoteAction, NoOpAction]


class RetailObservation(BaseModel):
    """Partial state visible to agent."""
    day: int
    cash: float
    
    # Inventory by product
    inventory: Dict[str, int]
    
    # Recent market signals (last 3 days avg)
    recent_demand_luxury: Dict[str, float]
    recent_demand_budget: Dict[str, float]
    recent_stockouts: Dict[str, int]  # Count by product
    
    # Current prices
    prices_luxury: Dict[str, float]
    prices_budget: Dict[str, float]
    competitor_prices: Dict[str, float]
    
    # Market conditions
    disruption_active: bool
    disruption_severity: float  # 0.0-1.0
    market_confidence: float    # 0.0-1.0 (agent hasn't seen true demand pattern)
    seasonality_multiplier: float  # Effect of current season/weekend on demand


class DisruptionEvent(BaseModel):
    """Represents a supply or demand shock."""
    event_type: str  # "demand_collapse", "supply_delay", "price_spike"
    product: str
    severity: float  # 0.0-1.0
    duration_days: int
    day_triggered: int


class RetailState(BaseModel):
    """Full internal state (hidden from agent except observation)."""
    day: int
    cash: float
    inventory: Dict[str, int]
    
    # Underlying demand patterns (hidden from agent)
    base_demand_luxury: Dict[str, float]
    base_demand_budget: Dict[str, float]
    demand_elasticity: Dict[str, float]  # Price sensitivity
    
    # Current prices by segment
    prices_luxury: Dict[str, float]
    prices_budget: Dict[str, float]
    competitor_prices: Dict[str, float]
    price_bounds: Dict[str, Dict[str, float]]  # min/max per product
    
    # Product economics
    product_costs: Dict[str, float]
    holding_costs: Dict[str, float]
    max_inventory: Dict[str, float]
    
    # Supply chain
    lead_time_mean: int
    lead_time_variance: int
    pending_orders: Dict[str, int]
    pending_order_queue: Dict[int, Dict[str, int]]
    supplier_reliability: float  # 0.0-1.0, chance orders arrive on time
    
    # Disruptions
    active_disruptions: List[DisruptionEvent]
    disruption_history: List[DisruptionEvent]
    next_disruption_day: Optional[int]
    
    # Metrics (for grading)
    cumulative_sales_luxury: Dict[str, float]
    cumulative_sales_budget: Dict[str, float]
    cumulative_revenue: float
    cumulative_holding_cost: float
    cumulative_stockouts: int
    cumulative_demand_lost: float

    # Realtime demand history (rolling window, last 3 steps per product)
    demand_history_luxury: Dict[str, List[float]] = Field(default_factory=dict)
    demand_history_budget: Dict[str, List[float]] = Field(default_factory=dict)

    # Per-product stockout counters (resets each episode)
    stockouts_per_product: Dict[str, int] = Field(default_factory=dict)
