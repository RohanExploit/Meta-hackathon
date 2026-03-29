import sys
sys.path.insert(0, '.')

from environment.retail_env import RetailInventoryEnv
from environment.models import RetailAction, ActionType, OrderAction
import json

def test_environment():
    print("Testing Retail Inventory Environment...")
    
    # Create environment
    env = RetailInventoryEnv(seed=42)
    
    # Define a simple task configuration
    task_config = {
        "products": ["Widget"],
        "initial_inventory": {"Widget": 10},
        "initial_cash": 100.0,
        "fixed_demand": {"Widget": 2},
        "product_costs": {"Widget": 5.0},
        "holding_costs": {"Widget": 0.1},
        "max_inventory": {"Widget": 100}
    }
    
    # Reset environment
    obs = env.reset(task_config)
    print(f"Initial observation: {obs}")
    
    # Test a few steps
    for i in range(3):
        print(f"\n--- Step {i+1} ---")
        
        # Create a simple action: order 2 widgets
        action = OrderAction(
            action=ActionType.ORDER,
            product="Widget",
            quantity=2
        )
        
        # Take step
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        if done:
            print("Episode finished!")
            break
    
    # Test state method
    state = env.get_state()
    print(f"\nFull state: {state}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_environment()