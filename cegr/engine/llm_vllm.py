import os
import re
from typing import Any, Callable, Dict, List, Optional

LLM_ENGINE = None
_MOCK_LLM_HANDLER: Optional[Callable[[List[Dict[str, str]]], str]] = None


_VLLM_INIT_ATTEMPTED = False


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
    global LLM_ENGINE, _VLLM_INIT_ATTEMPTED
    if LLM_ENGINE is not None:
        return LLM_ENGINE
    if _VLLM_INIT_ATTEMPTED:
        return None
    _VLLM_INIT_ATTEMPTED = True

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
        # Fallback smart mock handler if no vLLM and no explicit mock handler set
        results = []
        for msgs in messages_list:
            content_str = " ".join(m.get("content", "") for m in msgs)
            if "metaworld" in content_str.lower() or "hand_pos" in content_str:
                code_text = (
                    "```python\n"
                    "def reward_components(s, a, s2):\n"
                    "    c = {}\n"
                    "    c['step'] = -0.01\n"
                    "    d = float(np.linalg.norm(np.array(s2['hand_pos']) - np.array(s2['obj_pos'])))\n"
                    "    c['reach'] = -1.0 * d\n"
                    "    c['goal'] = 10.0 if s2.get('in_target', False) else 0.0\n"
                    "    return c\n"
                    "```"
                )
            else:
                code_text = (
                    "```python\n"
                    "def phi(st):\n"
                    "    return 0.5 * (1.0 if st.has_key else 0.0) + 0.5 * (1.0 if st.door == DOOR_OPEN else 0.0)\n"
                    "def reward_components(s, a, s2):\n"
                    "    c = {}\n"
                    "    c['step'] = -0.01\n"
                    "    c['shaping'] = phi(s2) - phi(s)\n"
                    "    c['goal'] = 1.0 if at_goal(s2) else 0.0\n"
                    "    return c\n"
                    "```"
                )
            results.append({
                "text": code_text,
                "prompt_tokens": len(content_str.split()),
                "completion_tokens": len(code_text.split()),
                "finish_reason": "stop",
            })
        return results

    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()

    # Determine maximum allowed input tokens to guarantee no VLLMValidationError
    model_max_len = 6144
    if hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "model_config"):
        model_max_len = getattr(engine.llm_engine.model_config, "max_model_len", 6144)
    max_input_tokens = max(512, model_max_len - max_tokens - 32)

    prompts = []
    for msgs in messages_list:
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        tok_ids = tokenizer.encode(p)
        if len(tok_ids) > max_input_tokens:
            p = tokenizer.decode(tok_ids[:max_input_tokens], skip_special_tokens=False)
        prompts.append(p)

    if seeds is not None and len(seeds) == len(prompts):
        params_list = [
            SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["```\n\n"], seed=int(sd))
            for sd in seeds
        ]
    else:
        seed_val = int(seeds[0]) if seeds else 42
        params_list = SamplingParams(temperature=temperature, max_tokens=max_tokens, stop=["```\n\n"], seed=seed_val)

    outputs = engine.generate(prompts, params_list, use_tqdm=False)
    results = []
    for out in outputs:
        results.append({
            "text": out.outputs[0].text,
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(out.outputs[0].token_ids),
            "finish_reason": out.outputs[0].finish_reason,
        })
    return results
