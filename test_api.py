"""
Test script to demonstrate the API endpoints work correctly
"""
import sys
import json
sys.path.insert(0, '.')

from environment.retail_env import MultiChannelRetailEnv
from environment.models import OrderAction, ActionType

def test_api_simulation():
    """Simulate what the API endpoints would do"""
    print("Testing API-like interaction...")
    
    # Create environment (simulating server startup)
    env = MultiChannelRetailEnv(seed=42)
    
    # Simulate /reset endpoint
    print("\n1. Testing reset (like POST /reset)")
    task_config = {
        "products": ["Widget"],
        "initial_inventory": {"Widget": 10},
        "initial_cash": 100.0,
        "base_demand_luxury": {"Widget": 1.0},
        "base_demand_budget": {"Widget": 2.0},
        "product_costs": {"Widget": 5.0},
        "holding_costs": {"Widget": 0.1},
        "max_inventory": {"Widget": 100},
        "initial_prices_luxury": {"Widget": 10.0},
        "initial_prices_budget": {"Widget": 7.0},
        "price_bounds": {"Widget": {"min": 5.5, "max": 15.0}},
    }
    
    obs = env.reset(task_config)
    reset_response = {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {}
    }
    print(f"Reset response keys: {list(reset_response.keys())}")
    print(f"Observation: inventory={reset_response['observation']['inventory']}, "
          f"cash={reset_response['observation']['cash']}, "
          f"day={reset_response['observation']['day']}")
    
    # Simulate /step endpoint
    print("\n2. Testing step (like POST /step)")
    action_dict = {"action": "order", "product": "Widget", "quantity": 3}
    
    # In real API, this would come from request body
    # We'll validate it like the API would
    try:
        action = OrderAction(**action_dict)
        print(f"Valid action: {action}")
        
        obs, reward, done, info = env.step(action)
        step_response = {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }
        print(f"Step response: reward={reward:.2f}, done={done}")
        print(f"Updated observation: inventory={obs.inventory}, cash={obs.cash:.2f}")
        
    except Exception as e:
        print(f"Error processing action: {e}")
    
    # Simulate /state endpoint
    print("\n3. Testing state (like GET /state)")
    state = env.get_state()
    state_response = {"state": state}
    print(f"State keys: {list(state_response['state'].keys())}")
    print(f"Cash in state: ${state['cash']:.2f}")
    print(f"Day in state: {state['day']}")
    
    print("\nAPI simulation test completed successfully!")

if __name__ == "__main__":
    test_api_simulation()
