from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ORDER = "order"
    SET_PRICE = "set_price"
    NO_OP = "noop"


class OrderAction(BaseModel):
    action: ActionType = ActionType.ORDER
    product: str
    quantity: int = Field(gt=0)


class SetPriceAction(BaseModel):
    action: ActionType = ActionType.SET_PRICE
    product: str
    new_price: float = Field(gt=0)


class NoOpAction(BaseModel):
    action: ActionType = ActionType.NO_OP


# Union type for all possible actions
RetailAction = Union[OrderAction, SetPriceAction, NoOpAction]


class RetailObservation(BaseModel):
    inventory: Dict[str, int]
    cash: float
    day: int
    sales_history: Dict[str, int]
    price: Dict[str, float]


class RetailState(BaseModel):
    inventory: Dict[str, int]
    cash: float
    day: int

    # Hidden market dynamics
    demand_pattern: Dict[str, float]  # base demand lambda per product
    demand_elasticity: Dict[str, float] = Field(default_factory=dict)
    reference_prices: Dict[str, float] = Field(default_factory=dict)
    seasonality: Dict[str, List[float]] = Field(default_factory=dict)

    # Cost and operations
    product_costs: Dict[str, float]
    product_holding_costs: Dict[str, float]
    max_inventory: Dict[str, float]
    lead_time: int = 0
    order_fixed_fee: float = 0.0
    price_bounds: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Fulfillment pipeline
    pending_orders: Dict[str, int] = Field(default_factory=dict)
    pending_order_queue: Dict[int, Dict[str, int]] = Field(default_factory=dict)

    # Current prices (partially exposed in observation)
    current_prices: Dict[str, float]

    # Optional cumulative episode metrics snapshot
    cumulative_metrics: Dict[str, float] = Field(default_factory=dict)

    # Optional metadata
    task_name: Optional[str] = None
    horizon: Optional[int] = None
