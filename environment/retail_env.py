"""Multi-channel retail environment with disruption recovery mechanics."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grader import score_episode
from .models import (
    AllocateAction,
    ActionType,
    CompositeAction,
    DisruptionEvent,
    NoOpAction,
    OrderAction,
    PromoteAction,
    RetailAction,
    RetailObservation,
    RetailState,
    SetPriceAction,
)


def _dump_model(model: Any) -> Dict[str, Any]:
    """Convert Pydantic model to dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class MultiChannelRetailEnv:
    """
    Dynamic multi-channel retail with disruption recovery.

    Agents manage:
    - Multi-segment pricing (luxury/budget customers)
    - Inventory allocation across segments
    - Supply chain disruptions (lead time, demand shocks)
    - Recovery strategies (promotions, dynamic pricing)
    - Multi-action timesteps (CompositeAction)
    - Full pipeline visibility (pending_orders in observation)
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.state: Optional[RetailState] = None
        self.initial_cash: float = 0.0
        self._horizon: int = 30
        self.episode_metrics: Dict[str, float] = {}

        # Disruption parameters
        self._disruption_probability: float = 0.15
        self._disruption_recovery_days: int = 5

    def reset(self, task_config: Dict[str, Any]) -> RetailObservation:
        """Reset environment with task configuration."""
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        products = list(task_config["products"])
        self._horizon = int(task_config.get("horizon", 30))

        # Initialize state
        inventory = {p: int(task_config["initial_inventory"].get(p, 5)) for p in products}
        cash = float(task_config["initial_cash"])
        self.initial_cash = cash

        # Demand patterns (hidden from agent)
        base_demand_luxury = {p: float(task_config.get("base_demand_luxury", {}).get(p, 1.5)) for p in products}
        base_demand_budget = {p: float(task_config.get("base_demand_budget", {}).get(p, 3.0)) for p in products}

        demand_elasticity = {p: float(task_config.get("demand_elasticity", {}).get(p, 1.2)) for p in products}

        # Economics
        product_costs = {p: float(task_config["product_costs"][p]) for p in products}
        holding_costs = {p: float(task_config.get("holding_costs", {}).get(p, 0.1)) for p in products}
        max_inventory = {p: float(task_config.get("max_inventory", {}).get(p, 50.0)) for p in products}

        # Pricing bounds
        price_bounds: Dict[str, Dict[str, float]] = {}
        for p in products:
            cost = product_costs[p]
            bounds_cfg = task_config.get("price_bounds", {}).get(p, {})
            price_bounds[p] = {
                "min": float(bounds_cfg.get("min", cost * 1.1)),
                "max": float(bounds_cfg.get("max", cost * 3.0)),
            }

        # Initial prices
        prices_luxury = {p: float(task_config.get("initial_prices_luxury", {}).get(p, product_costs[p] * 2.0)) for p in products}
        prices_budget = {p: float(task_config.get("initial_prices_budget", {}).get(p, product_costs[p] * 1.3)) for p in products}
        competitor_prices = {p: prices_budget[p] * 0.95 for p in products}

        # Supply chain
        lead_time_mean = int(task_config.get("lead_time_mean", 2))
        lead_time_variance = int(task_config.get("lead_time_variance", 1))
        supplier_reliability = float(task_config.get("supplier_reliability", 0.9))

        self.state = RetailState(
            day=0,
            cash=cash,
            inventory=inventory,
            base_demand_luxury=base_demand_luxury,
            base_demand_budget=base_demand_budget,
            demand_elasticity=demand_elasticity,
            prices_luxury=prices_luxury,
            prices_budget=prices_budget,
            competitor_prices=competitor_prices,
            price_bounds=price_bounds,
            product_costs=product_costs,
            holding_costs=holding_costs,
            max_inventory=max_inventory,
            lead_time_mean=lead_time_mean,
            lead_time_variance=lead_time_variance,
            pending_orders={p: 0 for p in products},
            pending_order_queue={},
            supplier_reliability=supplier_reliability,
            active_disruptions=[],
            disruption_history=[],
            next_disruption_day=None,
            cumulative_sales_luxury={p: 0.0 for p in products},
            cumulative_sales_budget={p: 0.0 for p in products},
            cumulative_revenue=0.0,
            cumulative_holding_cost=0.0,
            cumulative_stockouts=0,
            cumulative_demand_lost=0.0,
            cumulative_order_cost=0.0,
            demand_history_luxury={p: [] for p in products},
            demand_history_budget={p: [] for p in products},
            stockouts_per_product={p: 0 for p in products},
        )

        self.episode_metrics = {
            "horizon": float(self._horizon),
            "num_products": float(len(products)),
            "baseline_profit": float(task_config.get("baseline_profit", 100.0)),
            "total_steps": 0.0,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_holding_cost": 0.0,
            "total_order_cost": 0.0,
            "profit": 0.0,
            "total_demand": 0.0,
            "total_sales": 0.0,
            "fill_rate": 0.0,
            "stockout_count": 0,
            "disruption_events": 0,
            "recovery_success_rate": 0.0,
            "price_efficiency": 0.0,
            "max_possible_holding_cost": float(
                sum(float(max_inventory[p]) * float(holding_costs[p]) for p in products) * self._horizon
            ),
        }

        return self._get_observation()

    # ── Step (supports both single and composite actions) ───────────

    def step(self, action: RetailAction) -> Tuple[RetailObservation, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Supports both legacy single-action and new CompositeAction payloads.
        For CompositeAction, sub-actions are resolved in priority order:
            1. price_changes  (before demand calculation)
            2. orders         (supply chain)
            3. allocations    (inventory split)
            4. promotions     (demand boost)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        info: Dict[str, Any] = {
            "action_taken": _dump_model(action),
            "valid_action": True,
            "disruption_event": None,
            "composite": isinstance(action, CompositeAction),
        }

        # Check for new disruptions
        self._check_disruptions()

        # Apply disruption effects to demand
        disruption_multiplier = self._compute_disruption_multiplier()

        # Update competitor prices
        for p in self.state.inventory.keys():
            my_bdg = self.state.prices_budget[p]
            if self.state.competitor_prices[p] > my_bdg:
                self.state.competitor_prices[p] = max(self.state.product_costs[p] * 1.05, my_bdg * 0.95)
            else:
                self.state.competitor_prices[p] *= random.uniform(0.95, 1.05)

        # ── Process action(s) ───────────────────────────────────────
        action_reward = 0.0

        if isinstance(action, CompositeAction):
            # Multi-action timestep: execute in priority order
            sub_infos: List[Dict[str, Any]] = []

            # 1. Price changes first (affect demand calculation)
            for sub in (action.price_changes or []):
                sub_info: Dict[str, Any] = {"valid_action": True}
                action_reward += self._apply_set_price(sub, sub_info)
                sub_infos.append({"type": "set_price", **sub_info})

            # 2. Orders (supply chain)
            for sub in (action.orders or []):
                sub_info = {"valid_action": True}
                action_reward += self._apply_order(sub, sub_info)
                sub_infos.append({"type": "order", **sub_info})

            # 3. Allocations
            for sub in (action.allocations or []):
                sub_info = {"valid_action": True}
                action_reward += self._apply_allocate(sub, sub_info)
                sub_infos.append({"type": "allocate", **sub_info})

            # 4. Promotions
            for sub in (action.promotions or []):
                sub_info = {"valid_action": True}
                action_reward += self._apply_promote(sub, sub_info)
                sub_infos.append({"type": "promote", **sub_info})

            info["sub_actions"] = sub_infos
        else:
            # Legacy single-action path
            if action.action == ActionType.ALLOCATE:
                action_reward = self._apply_allocate(action, info)
            elif action.action == ActionType.SET_PRICE:
                action_reward = self._apply_set_price(action, info)
            elif action.action == ActionType.ORDER:
                action_reward = self._apply_order(action, info)
            elif action.action == ActionType.PROMOTE:
                action_reward = self._apply_promote(action, info)
            elif action.action == ActionType.NOOP:
                pass
            else:
                info["valid_action"] = False

        # Process pending order arrivals
        self._realize_arrivals()

        # Simulate market (demand and sales)
        market_reward = self._simulate_market(disruption_multiplier)

        # Apply holding costs
        holding_cost = self._apply_holding_costs()

        # Total reward (step-wise signal aligned with profit)
        total_reward = action_reward + market_reward - holding_cost

        # Update state
        self.state.day += 1
        done = self.state.day >= self._horizon or self.state.cash < -100.0

        # Finalize metrics
        if done:
            self._finalize_episode()
            grader = score_episode(self.episode_metrics)
            info["grader"] = grader
            info["terminal_summary"] = {
                "cash": float(self.state.cash),
                "profit": float(self.episode_metrics.get("profit", 0.0)),
                "metrics": {k: float(v) for k, v in self.episode_metrics.items()},
                "grader": grader,
            }

        self.episode_metrics["total_steps"] += 1.0

        observation = self._get_observation()
        return observation, total_reward, done, info

    # ── Disruption mechanics ────────────────────────────────────────

    def _check_disruptions(self) -> None:
        """Randomly trigger disruptions (supply/demand shocks)."""
        assert self.state is not None

        # Check if any active disruptions should end
        for disp in self.state.active_disruptions[:]:
            if self.state.day >= disp.day_triggered + disp.duration_days:
                self.state.active_disruptions.remove(disp)

        # Chance of new disruption
        if random.random() < self._disruption_probability:
            event_types = ["demand_collapse", "supply_delay", "demand_spike"]
            event_type = random.choice(event_types)
            product = random.choice(list(self.state.inventory.keys()))
            severity = random.uniform(0.3, 0.9)
            duration = random.randint(2, 5)

            disruption = DisruptionEvent(
                event_type=event_type,
                product=product,
                severity=severity,
                duration_days=duration,
                day_triggered=self.state.day,
            )
            self.state.active_disruptions.append(disruption)
            self.state.disruption_history.append(disruption)
            self.episode_metrics["disruption_events"] += 1.0

    def _compute_disruption_multiplier(self) -> float:
        """Compute demand multiplier based on active disruptions."""
        if not self.state.active_disruptions:
            return 1.0

        # Demand collapses reduce demand, spikes increase it
        multiplier = 1.0
        for disp in self.state.active_disruptions:
            if disp.event_type == "demand_collapse":
                multiplier *= (1.0 - disp.severity * 0.8)
            elif disp.event_type == "demand_spike":
                multiplier *= (1.0 + disp.severity * 1.5)

        return max(0.1, min(3.0, multiplier))  # Clamp extremes

    # ── Action handlers ─────────────────────────────────────────────

    def _apply_allocate(self, action: RetailAction, info: Dict[str, Any]) -> float:
        """Allocate inventory to luxury and budget segments."""
        assert self.state is not None

        if action.product not in self.state.inventory:
            info["valid_action"] = False
            return -1.0

        total_allocated = action.luxury_units + action.budget_units
        if total_allocated > self.state.inventory[action.product]:
            info["valid_action"] = False
            return -1.0

        # Store allocation for market simulation
        info["allocation_luxury"] = action.luxury_units
        info["allocation_budget"] = action.budget_units

        return 0.5  # Small reward for proactive allocation

    def _apply_set_price(self, action: RetailAction, info: Dict[str, Any]) -> float:
        """Set price for a segment."""
        assert self.state is not None

        if action.product not in self.state.price_bounds:
            info["valid_action"] = False
            return -1.0

        bounds = self.state.price_bounds[action.product]
        if action.new_price < bounds["min"] or action.new_price > bounds["max"]:
            info["valid_action"] = False
            return -1.0

        if action.segment == "luxury":
            self.state.prices_luxury[action.product] = action.new_price
        elif action.segment == "budget":
            self.state.prices_budget[action.product] = action.new_price
        else:
            info["valid_action"] = False
            return -1.0

        info["price_set"] = action.new_price
        return 0.0  # Neutral reward; actual benefit comes from sales

    def _apply_order(self, action: RetailAction, info: Dict[str, Any]) -> float:
        """Place an order with supplier (subject to reliability and lead time)."""
        assert self.state is not None

        if action.product not in self.state.inventory:
            info["valid_action"] = False
            return -2.0

        supplier_choice = getattr(action, "supplier", "A").upper()
        if supplier_choice == "B":
            # Fast, reliable, expensive
            unit_cost = self.state.product_costs[action.product] * 1.25
            lead_time_mean = 1
            lead_time_variance = 0
            reliability = 1.0
        else:
            # Slow, cheap, unreliable
            unit_cost = self.state.product_costs[action.product]
            lead_time_mean = self.state.lead_time_mean
            lead_time_variance = self.state.lead_time_variance
            reliability = self.state.supplier_reliability

        cost = float(action.quantity) * unit_cost
        if cost > self.state.cash:
            info["valid_action"] = False
            return -2.0

        # Deduct cost immediately
        self.state.cash -= cost
        self.state.cumulative_order_cost += cost
        self.episode_metrics["total_order_cost"] = self.state.cumulative_order_cost

        # Stochastic lead time
        lead_time = max(0, int(np.random.normal(lead_time_mean, lead_time_variance)))
        arrival_day = self.state.day + lead_time

        # Chance order doesn't arrive (supplier unreliability)
        if random.random() > reliability:
            info["order_lost"] = True
            return -1.0  # Penalty for lost order

        arrival_bucket = self.state.pending_order_queue.setdefault(arrival_day, {})
        arrival_bucket[action.product] = int(arrival_bucket.get(action.product, 0)) + action.quantity
        self.state.pending_orders[action.product] += action.quantity

        info["order_enqueued"] = True
        info["arrival_day"] = arrival_day
        info["lead_time"] = lead_time
        info["supplier"] = supplier_choice
        return 0.0  # Neutral; benefit comes later

    def _apply_promote(self, action: RetailAction, info: Dict[str, Any]) -> float:
        """Run a promotional campaign (increases demand at cost)."""
        assert self.state is not None

        if action.budget_allocated > self.state.cash:
            info["valid_action"] = False
            return -1.0

        self.state.cash -= action.budget_allocated
        info["promotion_budget"] = action.budget_allocated
        info["promotion_multiplier"] = 1.0 + (action.budget_allocated / 100.0)  # ROI estimate

        return -action.budget_allocated / 10.0  # Small upfront cost

    # ── Supply chain pipeline ───────────────────────────────────────

    def _realize_arrivals(self) -> None:
        """Process pending orders that arrive today."""
        assert self.state is not None

        due = self.state.pending_order_queue.pop(self.state.day, {})
        for product, qty in due.items():
            self.state.inventory[product] += int(qty)
            self.state.pending_orders[product] = max(
                0, int(self.state.pending_orders[product]) - int(qty)
            )

    def _build_pending_orders_observation(self) -> Dict[str, list]:
        """Build agent-visible pending order pipeline from internal queue.

        Transforms the internal {arrival_day: {product: qty}} queue into a
        per-product list of {quantity, days_to_arrival} dicts that satisfy
        the Markov property — the agent can now see its own pipeline.
        """
        assert self.state is not None

        products = list(self.state.inventory.keys())
        result: Dict[str, list] = {p: [] for p in products}

        for arrival_day, bucket in sorted(self.state.pending_order_queue.items()):
            days_to_arrival = max(0, arrival_day - self.state.day)
            for product, qty in bucket.items():
                if product in result:
                    result[product].append({
                        "quantity": int(qty),
                        "days_to_arrival": days_to_arrival,
                    })

        return result

    # ── Market simulation ───────────────────────────────────────────

    _DEMAND_HISTORY_WINDOW = 3  # rolling window size for demand history

    def _simulate_market(self, disruption_multiplier: float) -> float:
        """Simulate demand, allocations, and sales for each segment."""
        assert self.state is not None

        total_revenue = 0.0
        total_demand = 0.0
        total_sales = 0.0
        step_unmet = 0  # Track unmet demand for THIS step only

        seasonality_trend = 1.0 + 0.5 * np.sin(self.state.day / self._horizon * np.pi)

        for product in self.state.inventory.keys():
            comp_price = self.state.competitor_prices[product]
            # Base demand
            demand_lux = self._sample_demand(
                self.state.base_demand_luxury[product],
                self.state.prices_luxury[product],
                self.state.product_costs[product],
                self.state.demand_elasticity[product],
                disruption_multiplier * seasonality_trend,
                comp_price * 1.3,
            )
            demand_bdg = self._sample_demand(
                self.state.base_demand_budget[product],
                self.state.prices_budget[product],
                self.state.product_costs[product],
                self.state.demand_elasticity[product],
                disruption_multiplier * seasonality_trend,
                comp_price,
            )

            # Record actual demand in rolling history for realistic observations
            hist_lux = self.state.demand_history_luxury.setdefault(product, [])
            hist_lux.append(float(demand_lux))
            if len(hist_lux) > self._DEMAND_HISTORY_WINDOW:
                hist_lux.pop(0)

            hist_bdg = self.state.demand_history_budget.setdefault(product, [])
            hist_bdg.append(float(demand_bdg))
            if len(hist_bdg) > self._DEMAND_HISTORY_WINDOW:
                hist_bdg.pop(0)

            total_demand += demand_lux + demand_bdg

            # Allocate inventory (simple heuristic: luxury gets priority if priced higher)
            available = int(self.state.inventory[product])
            lux_price = self.state.prices_luxury[product]
            bdg_price = self.state.prices_budget[product]

            # Luxury segment gets preference if higher margin
            if lux_price > bdg_price:
                sales_lux = min(int(demand_lux), available // 2)
                sales_bdg = min(int(demand_bdg), available - sales_lux)
            else:
                sales_bdg = min(int(demand_bdg), available // 2)
                sales_lux = min(int(demand_lux), available - sales_bdg)

            # Revenue
            revenue_lux = sales_lux * lux_price
            revenue_bdg = sales_bdg * bdg_price
            total_revenue += revenue_lux + revenue_bdg

            # Update inventory and metrics
            self.state.inventory[product] -= sales_lux + sales_bdg
            self.state.cumulative_sales_luxury[product] += sales_lux
            self.state.cumulative_sales_budget[product] += sales_bdg
            total_sales += sales_lux + sales_bdg

            # Stockout tracking — per product
            unmet_lux = max(0, int(demand_lux) - sales_lux)
            unmet_bdg = max(0, int(demand_bdg) - sales_bdg)
            unmet = unmet_lux + unmet_bdg
            step_unmet += unmet

            if unmet > 0:
                self.episode_metrics["stockout_count"] += 1.0
                self.state.cumulative_stockouts += 1
                self.state.cumulative_demand_lost += float(unmet)
                self.state.stockouts_per_product[product] = (
                    self.state.stockouts_per_product.get(product, 0) + 1
                )

        self.state.cash += total_revenue
        self.state.cumulative_revenue += total_revenue

        self.episode_metrics["total_revenue"] += total_revenue
        self.episode_metrics["total_demand"] += total_demand
        self.episode_metrics["total_sales"] += total_sales

        # Return reward signal: per-step revenue minus per-step stockout penalty
        stockout_penalty = step_unmet * 2.0
        return total_revenue - stockout_penalty

    def _sample_demand(
        self,
        base_demand: float,
        price: float,
        product_cost: float,
        elasticity: float,
        disruption_multiplier: float,
        competitor_price: float,
    ) -> int:
        """Sample demand with price elasticity and disruptions."""
        rel_price = price / max(0.01, competitor_price)
        price_effect = rel_price ** (-elasticity * 1.5)

        effective_demand = base_demand * price_effect * disruption_multiplier
        effective_demand = max(0.1, effective_demand)

        return int(np.random.poisson(effective_demand))

    def _apply_holding_costs(self) -> float:
        """Apply inventory holding costs."""
        assert self.state is not None

        total_cost = 0.0
        for product, qty in self.state.inventory.items():
            cost = float(qty) * self.state.holding_costs[product]
            total_cost += cost

        self.state.cash -= total_cost
        self.state.cumulative_holding_cost += total_cost
        self.episode_metrics["total_cost"] += total_cost
        self.episode_metrics["total_holding_cost"] += total_cost

        return total_cost

    def _finalize_episode(self) -> None:
        """Finalize episode metrics."""
        assert self.state is not None

        profit = self.state.cash - self.initial_cash
        self.episode_metrics["profit"] = profit

        if self.episode_metrics["total_demand"] > 0:
            self.episode_metrics["fill_rate"] = self.episode_metrics["total_sales"] / self.episode_metrics["total_demand"]
        else:
            self.episode_metrics["fill_rate"] = 1.0

        if self.episode_metrics["disruption_events"] > 0:
            recovery_count = len([d for d in self.state.disruption_history if d.duration_days < 5])
            self.episode_metrics["recovery_success_rate"] = recovery_count / self.episode_metrics["disruption_events"]
        else:
            self.episode_metrics["recovery_success_rate"] = 0.0

    # ── Observation builder ─────────────────────────────────────────

    def _get_observation(self) -> RetailObservation:
        """Get partial state observation for agent (Markov-compliant)."""
        if self.state is None:
            raise RuntimeError("Environment not initialized")

        # Use actual rolling demand history when available; fall back to base constants
        # on day 0 before any market simulation has run.
        recent_demand_luxury: Dict[str, float] = {}
        recent_demand_budget: Dict[str, float] = {}
        for p in self.state.inventory.keys():
            hist_lux = self.state.demand_history_luxury.get(p, [])
            recent_demand_luxury[p] = (
                float(sum(hist_lux) / len(hist_lux))
                if hist_lux
                else float(max(0.1, self.state.base_demand_luxury[p]))
            )
            hist_bdg = self.state.demand_history_budget.get(p, [])
            recent_demand_budget[p] = (
                float(sum(hist_bdg) / len(hist_bdg))
                if hist_bdg
                else float(max(0.1, self.state.base_demand_budget[p]))
            )

        # Per-product stockout counts (how many steps this product had unmet demand)
        recent_stockouts = {
            p: int(self.state.stockouts_per_product.get(p, 0))
            for p in self.state.inventory.keys()
        }

        disruption_active = len(self.state.active_disruptions) > 0
        disruption_severity = max([d.severity for d in self.state.active_disruptions], default=0.0)
        market_confidence = max(0.0, 0.9 - (self.state.day / self._horizon) * 0.3)

        # Build pipeline visibility (Markov property compliance)
        pending_orders = self._build_pending_orders_observation()

        return RetailObservation(
            day=self.state.day,
            cash=self.state.cash,
            inventory=self.state.inventory.copy(),
            pending_orders=pending_orders,
            recent_demand_luxury=recent_demand_luxury,
            recent_demand_budget=recent_demand_budget,
            recent_stockouts=recent_stockouts,
            prices_luxury=self.state.prices_luxury.copy(),
            prices_budget=self.state.prices_budget.copy(),
            competitor_prices=self.state.competitor_prices.copy(),
            disruption_active=disruption_active,
            disruption_severity=disruption_severity,
            market_confidence=market_confidence,
            seasonality_multiplier=1.0 + 0.5 * np.sin(self.state.day / self._horizon * np.pi),
        )

    def get_state(self) -> Dict[str, Any]:
        """Get full internal state."""
        if self.state is None:
            raise RuntimeError("Environment not initialized")

        return _dump_model(self.state)
