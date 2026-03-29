from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grader import compute_deterministic_score
from .models import ActionType, RetailAction, RetailObservation, RetailState


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class RetailInventoryEnv:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.state: Optional[RetailState] = None
        self.last_actions: List[str] = []
        self.last_sales: Dict[str, int] = {}

        self.initial_cash: float = 0.0
        self._horizon: int = 30

        self._invalid_action_penalty: float = 1.0
        self._loop_penalty_value: float = 2.0
        self._service_penalty_value: float = 0.5
        self._price_volatility_penalty_weight: float = 0.0

        self.episode_metrics: Dict[str, float] = {}

    def reset(self, task_config: Dict[str, Any]) -> RetailObservation:
        """Reset environment with task-specific configuration."""

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        products = list(task_config["products"])

        self._horizon = int(task_config.get("horizon", 30))
        lead_time = max(0, int(task_config.get("lead_time", 0)))
        order_fixed_fee = float(task_config.get("order_fixed_fee", 0.0))
        self._invalid_action_penalty = float(task_config.get("invalid_action_penalty", 1.0))
        self._loop_penalty_value = float(task_config.get("loop_penalty", 2.0))
        self._service_penalty_value = float(task_config.get("service_penalty", 0.5))
        self._price_volatility_penalty_weight = float(
            task_config.get("price_volatility_penalty_weight", 0.0)
        )

        inventory = {
            product: int(task_config["initial_inventory"].get(product, 0)) for product in products
        }
        cash = float(task_config["initial_cash"])
        self.initial_cash = cash

        demand_pattern: Dict[str, float] = {}
        for product in products:
            if "fixed_demand" in task_config:
                demand_pattern[product] = float(task_config["fixed_demand"][product])
            elif "demand_ranges" in task_config:
                demand_min, demand_max = task_config["demand_ranges"][product]
                demand_pattern[product] = float(np.random.uniform(demand_min, demand_max))
            else:
                demand_pattern[product] = 1.0

        product_costs = {
            product: float(task_config["product_costs"][product]) for product in products
        }
        product_holding_costs = {
            product: float(task_config["holding_costs"][product]) for product in products
        }

        max_inventory = {
            product: float(task_config.get("max_inventory", {}).get(product, float("inf")))
            for product in products
        }

        demand_elasticity = {
            product: float(task_config.get("demand_elasticity", {}).get(product, 1.0))
            for product in products
        }

        seasonality_cfg = task_config.get("seasonality", {})
        seasonality = {
            product: list(seasonality_cfg.get(product, [1.0])) for product in products
        }

        reference_prices = {
            product: float(
                task_config.get("reference_prices", {}).get(
                    product, task_config["product_costs"][product] * 1.5
                )
            )
            for product in products
        }

        price_bounds_cfg = task_config.get("price_bounds", {})
        price_bounds: Dict[str, Dict[str, float]] = {}
        for product in products:
            cost = product_costs[product]
            product_bounds = price_bounds_cfg.get(product, {})
            min_multiplier = float(product_bounds.get("min_multiplier", 1.10))
            max_multiplier = float(product_bounds.get("max_multiplier", 2.00))
            min_price = float(product_bounds.get("min_price", cost * min_multiplier))
            max_price = float(product_bounds.get("max_price", cost * max_multiplier))
            if max_price < min_price:
                max_price = min_price
            price_bounds[product] = {"min": min_price, "max": max_price}

        initial_prices_cfg = task_config.get("initial_prices", {})
        current_prices: Dict[str, float] = {}
        for product in products:
            raw_price = float(initial_prices_cfg.get(product, reference_prices[product]))
            bounds = price_bounds[product]
            current_prices[product] = float(min(max(raw_price, bounds["min"]), bounds["max"]))

        pending_orders = {product: 0 for product in products}
        pending_order_queue: Dict[int, Dict[str, int]] = {}

        self.state = RetailState(
            inventory=inventory,
            cash=cash,
            day=0,
            demand_pattern=demand_pattern,
            demand_elasticity=demand_elasticity,
            reference_prices=reference_prices,
            seasonality=seasonality,
            product_costs=product_costs,
            product_holding_costs=product_holding_costs,
            pending_orders=pending_orders,
            pending_order_queue=pending_order_queue,
            max_inventory=max_inventory,
            current_prices=current_prices,
            lead_time=lead_time,
            order_fixed_fee=order_fixed_fee,
            price_bounds=price_bounds,
            cumulative_metrics={},
            task_name=task_config.get("name"),
            horizon=self._horizon,
        )

        self.last_actions = []
        self.last_sales = {product: 0 for product in products}

        per_step_max_holding = 0.0
        for product in products:
            cap = max_inventory[product]
            if cap == float("inf"):
                cap = max(inventory[product], 1)
            per_step_max_holding += float(cap) * product_holding_costs[product]

        self.episode_metrics = {
            "horizon": float(self._horizon),
            "num_products": float(len(products)),
            "target_profit": float(task_config.get("target_profit", task_config.get("baseline_profit", 1.0))),
            "baseline_profit": float(task_config.get("baseline_profit", 1.0)),
            "total_steps": 0.0,
            "total_demand": 0.0,
            "total_sales": 0.0,
            "total_unmet_demand": 0.0,
            "total_revenue": 0.0,
            "total_variable_order_cost": 0.0,
            "total_fixed_order_cost": 0.0,
            "total_order_cost": 0.0,
            "total_holding_cost": 0.0,
            "total_service_penalty": 0.0,
            "total_invalid_action_penalty": 0.0,
            "total_loop_penalty": 0.0,
            "total_price_volatility_penalty": 0.0,
            "total_penalties": 0.0,
            "price_change_count": 0.0,
            "price_change_magnitude": 0.0,
            "max_possible_holding_cost": max(0.0, per_step_max_holding * self._horizon),
            "ending_inventory_ratio": 0.0,
            "profit": 0.0,
            "price_change_budget_scale": 0.35,
        }

        self._sync_metrics_into_state()
        return self._get_observation()

    def step(self, action: RetailAction) -> Tuple[RetailObservation, float, bool, Dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._realize_due_arrivals()

        info: Dict[str, Any] = {
            "action_taken": _dump_model(action),
            "invalid_action": False,
            "arrivals": {},
        }

        step_revenue = 0.0
        step_variable_order_cost = 0.0
        step_fixed_order_cost = 0.0
        step_holding_cost = 0.0
        step_service_penalty = 0.0
        step_invalid_penalty = 0.0
        step_loop_penalty = 0.0
        step_price_volatility_penalty = 0.0

        action_str = self._action_signature(action)

        if self._detect_repeated_pattern(action_str):
            step_loop_penalty = self._loop_penalty_value
            info["loop_detected"] = True

        if action.action == ActionType.ORDER:
            valid_order, details = self._apply_order_action(action)
            info.update(details)
            if valid_order:
                step_variable_order_cost = float(details.get("variable_order_cost", 0.0))
                step_fixed_order_cost = float(details.get("fixed_order_cost", 0.0))
            else:
                info["invalid_action"] = True
                step_invalid_penalty += self._invalid_action_penalty
        elif action.action == ActionType.SET_PRICE:
            valid_price, details = self._apply_set_price_action(action)
            info.update(details)
            if valid_price:
                delta = float(details.get("price_change_abs", 0.0))
                self.episode_metrics["price_change_count"] += 1.0
                self.episode_metrics["price_change_magnitude"] += delta
                step_price_volatility_penalty = self._price_volatility_penalty_weight * delta
            else:
                info["invalid_action"] = True
                step_invalid_penalty += self._invalid_action_penalty
        elif action.action == ActionType.NO_OP:
            pass
        else:
            info["invalid_action"] = True
            step_invalid_penalty += self._invalid_action_penalty

        # Demand and sales execution
        sales: Dict[str, int] = {}
        demand_by_product: Dict[str, int] = {}
        unmet_by_product: Dict[str, int] = {}

        for product in self.state.inventory.keys():
            demand = self._sample_demand(product)
            demand_by_product[product] = demand

            available = int(self.state.inventory[product])
            sold = min(available, demand)
            unmet = max(0, demand - sold)

            sales[product] = sold
            unmet_by_product[product] = unmet

            price = float(self.state.current_prices[product])
            revenue = sold * price
            step_revenue += revenue

            self.state.inventory[product] = available - sold
            step_service_penalty += unmet * self._service_penalty_value

        self.last_sales = sales.copy()

        # Cash updates from market and carrying cost
        self.state.cash += step_revenue

        for product, qty in self.state.inventory.items():
            step_holding_cost += float(qty) * self.state.product_holding_costs[product]

        self.state.cash -= step_holding_cost

        step_penalties = (
            step_service_penalty
            + step_invalid_penalty
            + step_loop_penalty
            + step_price_volatility_penalty
        )

        reward = (
            step_revenue
            - step_variable_order_cost
            - step_fixed_order_cost
            - step_holding_cost
            - step_penalties
        )

        # Day increment and done conditions
        self.state.day += 1
        done = self.state.day >= self._horizon or self.state.cash < 0.0

        # Metrics update (single-source accounting)
        step_demand = float(sum(demand_by_product.values()))
        step_sales = float(sum(sales.values()))
        step_unmet = float(sum(unmet_by_product.values()))

        self.episode_metrics["total_steps"] += 1.0
        self.episode_metrics["total_demand"] += step_demand
        self.episode_metrics["total_sales"] += step_sales
        self.episode_metrics["total_unmet_demand"] += step_unmet
        self.episode_metrics["total_revenue"] += step_revenue
        self.episode_metrics["total_variable_order_cost"] += step_variable_order_cost
        self.episode_metrics["total_fixed_order_cost"] += step_fixed_order_cost
        self.episode_metrics["total_order_cost"] += step_variable_order_cost + step_fixed_order_cost
        self.episode_metrics["total_holding_cost"] += step_holding_cost
        self.episode_metrics["total_service_penalty"] += step_service_penalty
        self.episode_metrics["total_invalid_action_penalty"] += step_invalid_penalty
        self.episode_metrics["total_loop_penalty"] += step_loop_penalty
        self.episode_metrics["total_price_volatility_penalty"] += step_price_volatility_penalty
        self.episode_metrics["total_penalties"] += step_penalties

        self.last_actions.append(action_str)
        if len(self.last_actions) > 14:
            self.last_actions.pop(0)

        info.update(
            {
                "sales": sales,
                "demand": demand_by_product,
                "unmet_demand": unmet_by_product,
                "reward_components": {
                    "revenue": step_revenue,
                    "variable_order_cost": step_variable_order_cost,
                    "fixed_order_cost": step_fixed_order_cost,
                    "holding_cost": step_holding_cost,
                    "service_penalty": step_service_penalty,
                    "invalid_action_penalty": step_invalid_penalty,
                    "loop_penalty": step_loop_penalty,
                    "price_volatility_penalty": step_price_volatility_penalty,
                },
            }
        )

        if done:
            self._finalize_episode_summary(info)

        self._sync_metrics_into_state()
        info["episode_metrics"] = self._metrics_snapshot()

        observation = self._get_observation()
        return observation, reward, done, info

    def _action_signature(self, action: RetailAction) -> str:
        if action.action == ActionType.ORDER:
            return f"order:{action.product}:{action.quantity}"
        if action.action == ActionType.SET_PRICE:
            # bucket for repeated-pattern detection robustness to tiny wiggles
            bucket = round(float(action.new_price), 2)
            return f"set_price:{action.product}:{bucket}"
        return "noop"

    def _detect_repeated_pattern(self, current_signature: str) -> bool:
        if len(self.last_actions) < 3:
            return False

        recent = self.last_actions[-6:]

        # Direct repetition of same action
        if len(recent) >= 3 and recent[-1] == recent[-2] == recent[-3]:
            return True

        # Alternating two-action loops, e.g. A,B,A,B
        if len(recent) >= 4:
            pattern4 = recent[-4:]
            if pattern4[0] == pattern4[2] and pattern4[1] == pattern4[3] and pattern4[0] != pattern4[1]:
                return True

        # Dominant repetition in short window (micro-variation bypass resistance)
        extended = recent + [current_signature]
        top_count = Counter(extended).most_common(1)[0][1]
        if len(extended) >= 5 and top_count >= 4:
            return True

        return False

    def _realize_due_arrivals(self) -> None:
        assert self.state is not None
        due_day = int(self.state.day)
        due = self.state.pending_order_queue.pop(due_day, {})
        if not due:
            return

        for product, qty in due.items():
            self.state.inventory[product] += int(qty)
            self.state.pending_orders[product] = max(
                0, int(self.state.pending_orders.get(product, 0)) - int(qty)
            )

    def _enqueue_order(self, product: str, quantity: int) -> None:
        assert self.state is not None

        if self.state.lead_time == 0:
            self.state.inventory[product] += quantity
            return

        arrival_day = int(self.state.day) + int(self.state.lead_time)
        bucket = self.state.pending_order_queue.setdefault(arrival_day, {})
        bucket[product] = int(bucket.get(product, 0)) + int(quantity)
        self.state.pending_orders[product] = int(self.state.pending_orders.get(product, 0)) + int(quantity)

    def _apply_order_action(self, action: RetailAction) -> Tuple[bool, Dict[str, Any]]:
        assert self.state is not None

        details: Dict[str, Any] = {"order_enqueued": False}

        if action.product not in self.state.inventory:
            details["invalid_reason"] = "unknown_product"
            return False, details

        quantity = int(action.quantity)
        if quantity <= 0:
            details["invalid_reason"] = "non_positive_quantity"
            return False, details

        on_hand = int(self.state.inventory[action.product])
        in_transit = int(self.state.pending_orders.get(action.product, 0))
        cap = float(self.state.max_inventory[action.product])

        if on_hand + in_transit + quantity > cap:
            details["invalid_reason"] = "inventory_cap_exceeded"
            return False, details

        variable_order_cost = float(quantity) * self.state.product_costs[action.product]
        fixed_order_cost = float(self.state.order_fixed_fee)
        total_cost = variable_order_cost + fixed_order_cost

        if total_cost > float(self.state.cash):
            details["invalid_reason"] = "insufficient_cash"
            return False, details

        self.state.cash -= total_cost
        self._enqueue_order(action.product, quantity)

        details.update(
            {
                "order_enqueued": True,
                "order_arrival_day": int(self.state.day) + int(self.state.lead_time),
                "variable_order_cost": variable_order_cost,
                "fixed_order_cost": fixed_order_cost,
            }
        )
        return True, details

    def _apply_set_price_action(self, action: RetailAction) -> Tuple[bool, Dict[str, Any]]:
        assert self.state is not None

        details: Dict[str, Any] = {"price_updated": False}

        if action.product not in self.state.current_prices:
            details["invalid_reason"] = "unknown_product"
            return False, details

        new_price = float(action.new_price)
        bounds = self.state.price_bounds[action.product]
        if new_price < bounds["min"] or new_price > bounds["max"]:
            details["invalid_reason"] = "price_out_of_bounds"
            return False, details

        old_price = float(self.state.current_prices[action.product])
        self.state.current_prices[action.product] = new_price

        details.update(
            {
                "price_updated": True,
                "old_price": old_price,
                "new_price": new_price,
                "price_change_abs": abs(new_price - old_price),
            }
        )
        return True, details

    def _seasonality_factor(self, product: str, day: int) -> float:
        assert self.state is not None

        pattern = self.state.seasonality.get(product, [1.0])
        if not pattern:
            return 1.0
        return max(0.0, float(pattern[day % len(pattern)]))

    def _price_effect(self, product: str) -> float:
        assert self.state is not None

        reference_price = max(1e-6, float(self.state.reference_prices.get(product, 1.0)))
        current_price = max(1e-6, float(self.state.current_prices[product]))
        elasticity = max(0.0, float(self.state.demand_elasticity.get(product, 1.0)))

        # Smooth demand shrinkage as price rises relative to reference
        ratio = current_price / reference_price
        effect = ratio ** (-elasticity)

        # Clamp to avoid degenerate extremes while preserving directional signal
        return float(min(max(effect, 0.05), 3.0))

    def _sample_demand(self, product: str) -> int:
        assert self.state is not None

        base_lambda = max(0.0, float(self.state.demand_pattern.get(product, 0.0)))
        seasonality = self._seasonality_factor(product, self.state.day)
        price_effect = self._price_effect(product)
        effective_lambda = max(0.0, base_lambda * seasonality * price_effect)

        return int(np.random.poisson(effective_lambda))

    def _metrics_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for key, value in self.episode_metrics.items():
            snapshot[key] = float(value)
        return snapshot

    def _sync_metrics_into_state(self) -> None:
        if self.state is None:
            return
        self.state.cumulative_metrics = self._metrics_snapshot()

    def _finalize_episode_summary(self, info: Dict[str, Any]) -> None:
        assert self.state is not None

        inventory_value = 0.0
        max_capacity = 0.0
        used_capacity = 0.0
        for product, qty in self.state.inventory.items():
            inventory_value += float(qty) * self.state.product_costs[product]
            cap = float(self.state.max_inventory[product])
            if cap != float("inf"):
                max_capacity += cap
                used_capacity += float(qty)

        if max_capacity > 0:
            self.episode_metrics["ending_inventory_ratio"] = min(1.0, used_capacity / max_capacity)
        else:
            self.episode_metrics["ending_inventory_ratio"] = 0.0

        profit = float(self.state.cash) + inventory_value - self.initial_cash
        self.episode_metrics["profit"] = profit
        self.episode_metrics["horizon"] = float(self._horizon)
        self.episode_metrics["num_products"] = float(len(self.state.inventory))

        grader_out = compute_deterministic_score(self.episode_metrics)

        terminal_summary = {
            "cash": float(self.state.cash),
            "inventory_value": float(inventory_value),
            "profit": profit,
            "metrics": self._metrics_snapshot(),
            "grader": grader_out,
        }

        info["grader"] = grader_out
        info["terminal_summary"] = terminal_summary

    def _get_observation(self) -> RetailObservation:
        if self.state is None:
            raise RuntimeError("Environment not initialized")

        return RetailObservation(
            inventory=self.state.inventory.copy(),
            cash=float(self.state.cash),
            day=int(self.state.day),
            sales_history=self.last_sales.copy(),
            price=self.state.current_prices.copy(),
        )

    def get_state(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment not initialized")

        state_dict = _dump_model(self.state)
        state_dict["demand_pattern"] = {
            product: 0.0 for product in state_dict["demand_pattern"].keys()
        }
        return state_dict
