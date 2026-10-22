from typing import Any, Dict, List, Optional


def build_card_prompts(
    task_desc: str,
    api_desc: str,
    current_code: str,
    feedback_text: str,
    round_idx: int = 1,
) -> List[Dict[str, str]]:
    """Constructs prompt for CARD coder LLM."""
    system_prompt = (
        "You are an expert RL reward designer using Trajectory Preference Evaluation (CARD framework). "
        "Your task is to refine the reward function based on Process, Trajectory, and Preference feedback."
    )

    user_prompt = (
        f"{task_desc}\n\n"
        f"{api_desc}\n\n"
        f"--- CURRENT REWARD FUNCTION (Round {round_idx}) ---\n"
        f"```python\n{current_code.strip()}\n```\n\n"
        f"--- DYNAMIC FEEDBACK ---\n"
        f"{feedback_text}\n\n"
        "REPAIR INSTRUCTIONS:\n"
        "1. Fix the ranking inversions highlighted in the Preference Feedback so G(tau+) > G(tau-).\n"
        "2. Keep the reward dense with helpful sub-goal progress shaping.\n"
        "3. Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
