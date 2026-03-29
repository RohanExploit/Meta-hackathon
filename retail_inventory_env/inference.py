import os
import re
import textwrap
from typing import List, Dict, Any
import openai
import json

# Environment variables as mandated
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")  # Default for testing
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # Default for testing
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "test_token"  # Default for testing

# For testing purposes, we'll warn but not fail if using defaults
if API_BASE_URL == "https://api.openai.com/v1" and "OPENAI_API_KEY" not in os.environ:
    print("Warning: Using default API base URL. Set API_BASE_URL for custom endpoints.")
if MODEL_NAME == "gpt-3.5-turbo" and "MODEL_NAME" not in os.environ:
    print("Warning: Using default model name. Set MODEL_NAME environment variable.")
if HF_TOKEN == "test_token" and "HF_TOKEN" not in os.environ and "API_KEY" not in os.environ:
    print("Warning: Using default token. Set HF_TOKEN or API_KEY environment variable.")

# Configure OpenAI client for older version
openai.api_base = API_BASE_URL
openai.api_key = HF_TOKEN

# Inference parameters
MAX_STEPS = 30  # Match environment horizon
TEMPERATURE = 0.0  # Deterministic output
MAX_TOKENS = 100  # Enough for JSON action
FALLBACK_ACTION = {"action": "noop"}

# Regular expressions for parsing model output
ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"\{.*\}", re.DOTALL)  # Match JSON objects


def build_history_lines(history: List[str]) -> str:
    """Build history string from previous steps"""
    if not history:
        return "None"
    return "\n".join(history[-4:])


def extract_action_from_text(response_text: str) -> Dict[str, Any]:
    """Extract JSON action from model response"""
    if not response_text:
        return FALLBACK_ACTION.copy()

    # Prefer the first line that looks like a JSON action
    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            try:
                action_str = match.group(0).strip()
                action = json.loads(action_str)
                # Validate it's a proper action
                if isinstance(action, dict) and "action" in action:
                    return action
            except (json.JSONDecodeError, KeyError):
                continue

    # Fall back to searching the whole response
    match = ACTION_PATTERN.search(response_text)
    if match:
        try:
            action_str = match.group(0).strip()
            action = json.loads(action_str)
            if isinstance(action, dict) and "action" in action:
                return action
        except (json.JSONDecodeError, KeyError):
            pass

    return FALLBACK_ACTION.copy()


def build_system_prompt() -> str:
    """Build system prompt describing the task and valid actions"""
    return textwrap.dedent(
        """
        You are managing a retail store's inventory and pricing to maximize profit.
        You will receive observations about your store's state and must choose actions.
        
        Valid actions (respond with exactly one JSON object):
        1. {"action": "order", "product": "product_name", "quantity": integer}
           - Order more units of a product (must be positive integer)
           - Cannot order more than you can afford or exceed storage limits
        
        2. {"action": "set_price", "product": "product_name", "new_price": float}
           - Set the selling price for a product
           - Price must be between cost+10% and cost+100%
        
        3. {"action": "noop"}
           - Take no action this step
        
        Your goal is to maximize profit over the episode by balancing:
        - Revenue from sales
        - Cost of ordering inventory
        - Holding costs for inventory
        - Avoiding stockouts (lost sales)
        
        Respond with ONLY the JSON action object, no additional text.
        """
    ).strip()


def build_user_prompt(step: int, observation: Dict[str, Any], history: List[str]) -> str:
    """Build user prompt with current observation and history"""
    # Format observation clearly
    obs_lines = []
    obs_lines.append(f"Day: {observation['day']}")
    obs_lines.append(f"Cash: ${observation['cash']:.2f}")
    obs_lines.append("Inventory:")
    for product, quantity in observation['inventory'].items():
        price = observation['price'].get(product, 0)
        obs_lines.append(f"  {product}: {quantity} units (selling at ${price:.2f})")
    obs_lines.append("Recent sales (yesterday):")
    for product, quantity in observation['sales_history'].items():
        obs_lines.append(f"  {product}: {quantity} units")
    
    observation_text = "\n".join(obs_lines)
    
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        
        Current Store State:
        {observation_text}
        
        Previous steps:
        {build_history_lines(history)}
        
        Choose your next action to maximize profit.
        Respond with exactly one JSON action object.
        """
    ).strip()
    
    return prompt


def main() -> None:
    """Main inference loop"""
    print("Starting Retail Inventory Environment Inference")
    print(f"Using model: {MODEL_NAME}")
    print(f"API Base URL: {API_BASE_URL}")
    
    # In a real submission, the evaluation system would handle reset/step calls
    # This script demonstrates how an agent would interact with the environment
    
    # Example usage (would be replaced by actual evaluation loop):
    print("\nThis script shows how to interact with the environment.")
    print("In practice, the evaluation system would:")
    print("1. Call /reset to initialize the environment")
    print("2. Repeatedly call /step with actions generated by this script")
    print("3. Use the rewards and observations to guide future actions")
    
    # Example of how to generate an action:
    example_observation = {
        "inventory": {"product_a": 10, "product_b": 5},
        "cash": 100.0,
        "day": 0,
        "sales_history": {"product_a": 0, "product_b": 0},
        "price": {"product_a": 15.0, "product_b": 8.0}
    }
    
    example_history = []
    user_prompt = build_user_prompt(0, example_observation, example_history)
    system_prompt = build_system_prompt()
    
    print("\nExample interaction:")
    print(f"System Prompt: {system_prompt[:100]}...")
    print(f"User Prompt: {user_prompt[:100]}...")
    
    # In actual use, we would call the model here:
    # For openai v0.27.10
    # completion = openai.ChatCompletion.create(
    #     model=MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ],
    #     temperature=TEMPERATURE,
    #     max_tokens=MAX_TOKENS
    # )
    # response_text = completion.choices[0].message.content
    # action = extract_action_from_text(response_text)
    # print(f"Generated action: {action}")


if __name__ == "__main__":
    main()