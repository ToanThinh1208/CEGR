import os
import re
from typing import Any, Callable, Dict, List, Optional

LLM_ENGINE = None
_MOCK_LLM_HANDLER: Optional[Callable[[List[Dict[str, str]]], str]] = None


def set_mock_llm(handler: Optional[Callable[[List[Dict[str, str]]], str]]):
    """Sets a deterministic mock handler for local tests and CPU debugging without vLLM."""
    global _MOCK_LLM_HANDLER
    _MOCK_LLM_HANDLER = handler


def init_llm(
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    max_model_len: int = 6144,
    gpu_memory_utilization: float = 0.88,
    tensor_parallel_size: int = 2,
):
    global LLM_ENGINE
    if LLM_ENGINE is not None:
        return LLM_ENGINE

    try:
        import torch
        from vllm import LLM

        actual_tp = max(1, min(tensor_parallel_size, torch.cuda.device_count()))
        print(f"Loading {model_id} onto GPU (TP={actual_tp}) ...")
        LLM_ENGINE = LLM(
            model=model_id,
            tensor_parallel_size=actual_tp,
            dtype="half",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        print("✓ vLLM Engine successfully initialized!")
        return LLM_ENGINE
    except Exception as e:
        print(f"[Warning] Could not initialize vLLM: {e}. Running in Mock / CPU fallback mode.")
        return None


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if m:
        return max(m, key=len).strip()
    if "def reward_components" in text:
        return text[text.index("def reward_components"):].strip()
    return text.strip()


def llm_batch_chat(
    messages_list: List[List[Dict[str, str]]],
    temperature: float = 0.35,
    max_tokens: int = 900,
    seeds: Optional[List[int]] = None,
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
) -> List[Dict[str, Any]]:
    """
    Executes parallel batch inference for hundreds of prompts in a single vLLM pass.
    Falls back gracefully to mock handler if running offline or in unit tests.
    """
    if not messages_list:
        return []

    # If mock handler active (e.g. during testing)
    if _MOCK_LLM_HANDLER is not None:
        results = []
        for msgs in messages_list:
            text = _MOCK_LLM_HANDLER(msgs)
            results.append({
                "text": text,
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in msgs),
                "completion_tokens": len(text.split()),
                "finish_reason": "stop",
            })
        return results

    engine = init_llm(model_id=model_id)
    if engine is None:
        # Fallback dummy code if no vLLM and no mock handler set
        default_code = (
            "```python\ndef reward_components(s, a, s2):\n"
            "    c = {}\n"
            "    c['step'] = -0.01\n"
            "    c['goal'] = 1.0 if at_goal(s2) else 0.0\n"
            "    return c\n```"
        )
        return [
            {
                "text": default_code,
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "finish_reason": "stop",
            }
            for _ in messages_list
        ]

    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()
    prompts = []
    for msgs in messages_list:
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    if seeds is not None and len(seeds) == len(prompts):
        params_list = [
            SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["```\n\n"], seed=int(sd))
            for sd in seeds
        ]
    else:
        seed_val = int(seeds[0]) if seeds else 42
        params_list = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["```\n\n"], seed=seed_val)

    outputs = engine.generate(prompts, params_list, use_tqdm=True)
    results = []
    for out in outputs:
        results.append({
            "text": out.outputs[0].text,
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(out.outputs[0].token_ids),
            "finish_reason": out.outputs[0].finish_reason,
        })
    return results
