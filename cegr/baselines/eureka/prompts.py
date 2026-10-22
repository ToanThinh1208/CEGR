from typing import Any, Dict, List, Optional


def build_eureka_prompts(
    current_code: str,
    reflection_text: str,
    task_desc: str,
    api_desc: str,
    generation: int = 1,
) -> List[Dict[str, str]]:
    """Constructs Eureka evolutionary mutation prompts for LLM."""
    system_prompt = (
        "You are an expert in reinforcement learning reward design. Your objective is to design "
        "and mutate dense reward functions that guide the policy to solve the task efficiently."
    )

    user_prompt = (
        f"{task_desc}\n\n"
        f"{api_desc}\n\n"
        f"--- CURRENT REWARD FUNCTION (Generation {generation}) ---\n"
        f"```python\n{current_code.strip()}\n```\n\n"
        f"--- TRAINING REFLECTION ---\n"
        f"{reflection_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Write a newly mutated, improved `reward_components(s, a, s2)` in Python.\n"
        "2. Ensure shaping components provide smooth learning signals without dominating the terminal goal.\n"
        "3. Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
