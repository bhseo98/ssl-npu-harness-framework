"""LLM implementations.

Built-in: `hf_causal` (any HuggingFace causal LM with a chat template).
Default model in config = Qwen/Qwen2.5-0.5B-Instruct (~1GB fp16, Korean).
"""
from __future__ import annotations

from .interfaces import BaseLLM
from .registry import register


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


_END_OF_TURN_CANDIDATES = (
    "<|endofturn|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
)
_LEAK_MARKERS = ("\nassistant", "\nuser", "\nsystem", "<|endofturn|>", "<|im_end|>")


@register("llm", "hf_causal")
class HFCausalLM(BaseLLM):
    def __init__(
        self,
        model: str,
        device: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        repetition_penalty: float = 1.15,
        system_prompt: str = "",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = _resolve_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._system_prompt = system_prompt

        # Collect chat-template end-of-turn ids that exist in vocab.
        # Without these, many instruct models keep emitting role markers past their turn.
        unk = self._tokenizer.unk_token_id
        eos_ids = {self._tokenizer.eos_token_id}
        for tok in _END_OF_TURN_CANDIDATES:
            tid = self._tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != unk:
                eos_ids.add(tid)
        self._eos_ids = [t for t in eos_ids if t is not None]

    def chat(
        self,
        user_message: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        import torch

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        for u, a in history or []:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_message})

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0,
                temperature=max(self._temperature, 1e-5),
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._eos_ids,
                repetition_penalty=self._repetition_penalty,
            )

        gen = output[0][inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        for marker in _LEAK_MARKERS:
            idx = text.find(marker)
            if idx >= 0:
                text = text[:idx].rstrip()
        return text
