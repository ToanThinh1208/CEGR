"""
Eureka prompt templates and builders for CEGR.
Adapted directly from Eureka (Ma et al., ICLR 2024): eureka/utils/prompts/
  - initial_system.txt
  - code_output_tip.txt
  - initial_user.txt
  - reward_signature.txt
  - policy_feedback.txt
  - code_feedback.txt
  - execution_error_feedback.txt
"""
from typing import Any, Dict, List, Optional


# 1. initial_system (eureka/utils/prompts/initial_system.txt)
INITIAL_SYSTEM = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
Your reward function should use useful variables from the environment as inputs. As an example,
the reward function signature can be:
{task_reward_signature_string}
""".strip()

# 2. reward_signature (eureka/utils/prompts/reward_signature.txt adapted for CEGR environment interface)
REWARD_SIGNATURE = """
def reward_components(s, a, s2) -> Dict[str, float]:
    # Returns a dictionary of each individual reward component
    return {"step": -0.01, "shaping": 0.0, "goal": 1.0}
""".strip()

# 3. code_output_tip (eureka/utils/prompts/code_output_tip.txt)
CODE_OUTPUT_TIP = """
The output of the reward function should consist of a dictionary of individual reward components (the total reward is the sum of its values).
The code output should be formatted as a python code string: "```python ... ```".

Some helpful tips for writing the reward function code:
    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations (e.g. math.exp, clip, or difference shaping) to the overall reward or its components.
    (2) If you choose to transform a reward component, then you must also introduce a temperature or scale parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable.
    (3) Ensure shaping components provide smooth learning signals without dominating the terminal goal.
    (4) Most importantly, the reward code's input variables must contain only attributes of the provided environment specification (s, a, s2). Under no circumstance can you introduce new input variables or nonexistent attributes.
""".strip()

# 4. initial_user (eureka/utils/prompts/initial_user.txt)
INITIAL_USER = """
The Python environment specification is:
{api_desc}

Write a reward function for the following task:
{task_desc}
""".strip()

# 5. policy_feedback (eureka/utils/prompts/policy_feedback.txt)
POLICY_FEEDBACK = """
We trained an RL policy using your provided reward function code and tracked the values of each reward component as well as policy metrics (success rates) across training checkpoints:
""".strip()

# 6. code_feedback (eureka/utils/prompts/code_feedback.txt)
CODE_FEEDBACK = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component 
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""".strip()

# 7. execution_error_feedback (eureka/utils/prompts/execution_error_feedback.txt)
EXECUTION_ERROR_FEEDBACK = """
Executing the reward function code above has the following error: {traceback_msg}. Please fix the bug and provide a new, improved reward function!
""".strip()

# Aliases mirroring Eureka lowercase filenames
initial_system = INITIAL_SYSTEM
reward_signature = REWARD_SIGNATURE
code_output_tip = CODE_OUTPUT_TIP
initial_user = INITIAL_USER
policy_feedback = POLICY_FEEDBACK
code_feedback = CODE_FEEDBACK
execution_error_feedback = EXECUTION_ERROR_FEEDBACK


def build_initial_eureka_prompts(
    task_desc: str,
    api_desc: str,
    reward_sig: str = REWARD_SIGNATURE,
) -> List[Dict[str, str]]:
    """Constructs initial (Generation 0) Eureka prompt before any training reflection."""
    system_prompt = (
        INITIAL_SYSTEM.format(task_reward_signature_string=reward_sig)
        + "\n\n"
        + CODE_OUTPUT_TIP
    )
    user_prompt = INITIAL_USER.format(api_desc=api_desc, task_desc=task_desc)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_eureka_prompts(
    current_code: str,
    reflection_text: str,
    task_desc: str,
    api_desc: str,
    generation: int = 1,
    execution_error: Optional[str] = None,
    reward_sig: str = REWARD_SIGNATURE,
) -> List[Dict[str, str]]:
    """
    Constructs Eureka evolutionary mutation prompts for LLM, following the Eureka prompt architecture.
    """
    system_prompt = (
        INITIAL_SYSTEM.format(task_reward_signature_string=reward_sig)
        + "\n\n"
        + CODE_OUTPUT_TIP
    )

    user_parts = [
        f"{task_desc}\n\n{api_desc}",
        f"--- CURRENT REWARD FUNCTION (Generation {generation}) ---\n```python\n{current_code.strip()}\n```",
    ]

    if execution_error:
        user_parts.append(
            f"--- EXECUTION ERROR ---\n"
            + EXECUTION_ERROR_FEEDBACK.format(traceback_msg=execution_error)
        )

    if reflection_text and reflection_text.strip():
        user_parts.append(
            f"--- TRAINING REFLECTION ---\n"
            f"{POLICY_FEEDBACK}\n"
            f"{reflection_text.strip()}"
        )

    user_parts.append(f"--- FEEDBACK ANALYSIS & MUTATION GUIDANCE ---\n{CODE_FEEDBACK}")

    user_parts.append(
        "INSTRUCTIONS:\n"
        "1. Write a newly mutated, improved `reward_components(s, a, s2)` in Python.\n"
        "2. Ensure shaping components provide smooth learning signals without dominating the terminal goal.\n"
        "3. Reply with ONLY one Python code block defining `reward_components(s, a, s2)`."
    )

    user_prompt = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


__all__ = [
    "build_eureka_prompts",
    "build_initial_eureka_prompts",
    "INITIAL_SYSTEM",
    "REWARD_SIGNATURE",
    "CODE_OUTPUT_TIP",
    "INITIAL_USER",
    "POLICY_FEEDBACK",
    "CODE_FEEDBACK",
    "EXECUTION_ERROR_FEEDBACK",
    "initial_system",
    "reward_signature",
    "code_output_tip",
    "initial_user",
    "policy_feedback",
    "code_feedback",
    "execution_error_feedback",
]
